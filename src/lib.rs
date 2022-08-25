use std::{
    collections::HashMap,
    fs::{self, File},
    io::{self, BufRead, BufReader, Error, ErrorKind, Write},
    ops::Deref,
    path::Path,
    str::FromStr,
    string::String,
};

use chrono::{DateTime, TimeZone, Utc};
use lazy_static::lazy_static;
use regex::{Regex, RegexSet};
use serde::{Deserialize, Serialize};

// ref. https://github.com/cmars/prometheus-scrape/blob/master/src/lib.rs
// ref. https://github.com/ccakes/prometheus-parse-rs/blob/master/src/lib.rs
lazy_static! {
    static ref HELP_RE: Regex = Regex::new(r"^#\s+HELP\s+(\w+)\s+(.+)$").unwrap();
    static ref TYPE_RE: Regex = Regex::new(r"^#\s+TYPE\s+(\w+)\s+(\w+)").unwrap();
    static ref METRIC_RE: Regex = Regex::new(
        r"^(?P<name>\w+)(\{(?P<labels>[^}]+)\})?\s+(?P<value>\S+)(\s+(?P<timestamp>\S+))?"
    )
    .unwrap();
}

#[derive(Debug, Eq, PartialEq)]
pub enum Entry<'a> {
    Doc {
        metric_name: &'a str,
        doc: &'a str,
    },
    Type {
        metric_name: String,
        sample_type: MetricKind,
    },
    Metric {
        metric_name: &'a str,
        labels: Option<&'a str>,
        value: &'a str,
        timestamp: Option<&'a str>,
    },
    Empty,
    Ignored,
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum MetricKind {
    Counter,
    Gauge,
    Histogram,
    Summary,
    Untyped,
}

/// ref. https://doc.rust-lang.org/std/str/trait.FromStr.html
impl FromStr for MetricKind {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let decoded = match s {
            "counter" => MetricKind::Counter,
            "gauge" => MetricKind::Gauge,
            "histogram" => MetricKind::Histogram,
            "summary" => MetricKind::Summary,
            _ => MetricKind::Untyped,
        };
        Ok(decoded)
    }
}

impl<'a> Entry<'a> {
    pub fn parse_line(line: &'a str) -> Entry<'a> {
        let line = line.trim();
        if line.is_empty() {
            return Entry::Empty;
        }

        if let Some(ref caps) = HELP_RE.captures(line) {
            return match (caps.get(1), caps.get(2)) {
                (Some(ref metric_name), Some(ref doc)) => Entry::Doc {
                    metric_name: metric_name.as_str(),
                    doc: doc.as_str(),
                },
                _ => Entry::Ignored,
            };
        }

        if let Some(ref caps) = TYPE_RE.captures(line) {
            return match (caps.get(1), caps.get(2)) {
                (Some(ref metric_name), Some(ref sample_type)) => {
                    let sample_type = MetricKind::from_str(sample_type.as_str()).unwrap();
                    Entry::Type {
                        metric_name: match sample_type {
                            MetricKind::Histogram => format!("{}_bucket", metric_name.as_str()),
                            _ => metric_name.as_str().to_string(),
                        },
                        sample_type,
                    }
                }
                _ => Entry::Ignored,
            };
        }

        if let Some(ref caps) = METRIC_RE.captures(line) {
            return match (
                caps.name("name"),
                caps.name("labels"),
                caps.name("value"),
                caps.name("timestamp"),
            ) {
                (Some(ref name), labels, Some(ref value), timestamp) => Entry::Metric {
                    metric_name: name.as_str(),
                    labels: labels.map(|c| c.as_str()),
                    value: value.as_str(),
                    timestamp: timestamp.map(|c| c.as_str()),
                },
                _ => Entry::Ignored,
            };
        } else {
            Entry::Ignored
        }
    }
}

/// RUST_LOG=debug cargo test --all-features --package prometheus-manager --lib -- test_lineinfo_parse --exact --show-output
#[test]
fn test_lineinfo_parse() {
    assert_eq!(
        Entry::parse_line("foo 2"),
        Entry::Metric {
            metric_name: "foo",
            value: "2",
            labels: None,
            timestamp: None,
        }
    );
    assert_eq!(
        Entry::parse_line("foo wtf -1"),
        Entry::Metric {
            metric_name: "foo",
            value: "wtf",
            labels: None,
            timestamp: Some("-1"),
        }
    );
    assert_eq!(Entry::parse_line("foo=2"), Entry::Ignored,);
    assert_eq!(
        Entry::parse_line("foo 2 1543182234"),
        Entry::Metric {
            metric_name: "foo",
            value: "2",
            labels: None,
            timestamp: Some("1543182234"),
        }
    );
    assert_eq!(
        Entry::parse_line("foo{bar=baz} 2 1543182234"),
        Entry::Metric {
            metric_name: "foo",
            value: "2",
            labels: Some("bar=baz"),
            timestamp: Some("1543182234"),
        }
    );
    assert_eq!(
        Entry::parse_line("foo{bar=baz,quux=nonce} 2 1543182234"),
        Entry::Metric {
            metric_name: "foo",
            value: "2",
            labels: Some("bar=baz,quux=nonce"),
            timestamp: Some("1543182234"),
        }
    );
    assert_eq!(
        Entry::parse_line("# HELP foo this is a docstring"),
        Entry::Doc {
            metric_name: "foo",
            doc: "this is a docstring"
        },
    );
    assert_eq!(
        Entry::parse_line("# TYPE foobar bazquux"),
        Entry::Type {
            metric_name: "foobar".to_string(),
            sample_type: MetricKind::Untyped,
        },
    );
}

#[derive(Debug, PartialEq, Clone)]
pub struct Metric {
    pub metric: String,
    pub value: Value,
    pub labels: Option<Labels>,
    pub timestamp: Option<DateTime<Utc>>,
}

impl Default for Metric {
    fn default() -> Self {
        Self::default()
    }
}

impl Metric {
    pub fn default() -> Self {
        Self {
            metric: String::new(),
            value: Value::Untyped(0.0),
            labels: None,
            timestamp: None,
        }
    }

    /// Stringify the metric name with labels (if any).
    pub fn name_with_labels(&self) -> String {
        if let Some(labels) = &self.labels {
            let mut pairs: Vec<String> = Vec::with_capacity(labels.len());
            for (k, v) in labels.iter() {
                pairs.push(format!("{}_{}", k.replace(' ', ""), v.replace(' ', "")));
            }
            // sort in lexicographic increasing order
            pairs.sort();
            format!("{}_{}", self.metric, pairs.join("_"))
        } else {
            self.metric.clone()
        }
    }
}

/// RUST_LOG=debug cargo test --all-features --package prometheus-manager --lib -- test_metric_name_with_labels --exact --show-output
#[test]
fn test_metric_name_with_labels() {
    assert_eq!(
        "http_requests_total",
        &Metric {
            metric: "http_requests_total".to_string(),
            ..Default::default()
        }
        .name_with_labels(),
    );
    assert_eq!(
        "http_requests_total_code_400_method_post",
        &Metric {
            metric: "http_requests_total".to_string(),
            labels: Some(Labels(
                [("method", "post"), ("code", "400")]
                    .iter()
                    .map(pair_to_string)
                    .collect()
            )),
            ..Default::default()
        }
        .name_with_labels(),
    );
    assert_eq!(
        "http_requests_total_code_400_method_postpost",
        &Metric {
            metric: "http_requests_total".to_string(),
            labels: Some(Labels(
                [("method", "post    post"), ("code", "400")]
                    .iter()
                    .map(pair_to_string)
                    .collect()
            )),
            ..Default::default()
        }
        .name_with_labels(),
    );
}

fn parse_bucket(s: &str, label: &str) -> Option<f64> {
    if let Some(kv) = s.split(',').next() {
        let kvpair = kv.split('=').collect::<Vec<_>>();
        let (k, v) = (kvpair[0], kvpair[1].trim_matches('"'));
        if k == label {
            match parse_golang_float(v) {
                Ok(v) => Some(v),
                Err(_) => None,
            }
        } else {
            None
        }
    } else {
        None
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct HistogramCount {
    pub less_than: f64,
    pub count: f64,
}

#[derive(Debug, PartialEq, Clone)]
pub struct SummaryCount {
    pub quantile: f64,
    pub count: f64,
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct Labels(HashMap<String, String>);

impl Labels {
    fn from_str(s: &str) -> Labels {
        let mut l = HashMap::new();
        for kv in s.split(',') {
            let kvpair = kv.split('=').collect::<Vec<_>>();
            if kvpair.len() != 2 || kvpair[0].is_empty() {
                continue;
            }
            l.insert(
                kvpair[0].to_string(),
                kvpair[1].trim_matches('"').to_string(),
            );
        }
        Labels(l)
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.0.get(name).map(|x| x.as_str())
    }
}

impl Deref for Labels {
    type Target = HashMap<String, String>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// RUST_LOG=debug cargo test --all-features --package prometheus-manager --lib -- test_labels_parse --exact --show-output
#[test]
fn test_labels_parse() {
    assert_eq!(
        Labels::from_str("foo=bar"),
        Labels([("foo", "bar")].iter().map(pair_to_string).collect())
    );
    assert_eq!(
        Labels::from_str("foo=bar,"),
        Labels([("foo", "bar")].iter().map(pair_to_string).collect())
    );
    assert_eq!(
        Labels::from_str(",foo=bar,"),
        Labels([("foo", "bar")].iter().map(pair_to_string).collect())
    );
    assert_eq!(
        Labels::from_str("=,foo=bar,"),
        Labels([("foo", "bar")].iter().map(pair_to_string).collect())
    );
    assert_eq!(
        Labels::from_str(r#"foo="bar""#),
        Labels([("foo", "bar")].iter().map(pair_to_string).collect())
    );
    assert_eq!(
        Labels::from_str(r#"foo="bar",baz="quux""#),
        Labels(
            [("foo", "bar"), ("baz", "quux")]
                .iter()
                .map(pair_to_string)
                .collect()
        )
    );
    assert_eq!(
        Labels::from_str(r#"foo="foo bar",baz="baz quux""#),
        Labels(
            [("foo", "foo bar"), ("baz", "baz quux")]
                .iter()
                .map(pair_to_string)
                .collect()
        )
    );
    assert_eq!(Labels::from_str("==="), Labels(HashMap::new()),);
}

#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Counter(f64),
    Gauge(f64),
    Histogram(Vec<HistogramCount>),
    Summary(Vec<SummaryCount>),
    Untyped(f64),
}

impl Value {
    pub fn to_f64(&self) -> f64 {
        match self {
            Value::Counter(v) => *v,
            Value::Gauge(v) => *v,
            Value::Untyped(v) => *v,

            // TODO: fix this
            Value::Histogram(_) => 0.0,
            Value::Summary(_) => 0.0,
        }
    }

    fn push_histogram(&mut self, h: HistogramCount) {
        if let &mut Value::Histogram(ref mut hs) = self {
            hs.push(h)
        }
    }

    fn push_summary(&mut self, s: SummaryCount) {
        if let &mut Value::Summary(ref mut ss) = self {
            ss.push(s)
        }
    }
}

#[derive(Debug)]
pub struct Scrape {
    pub docs: HashMap<String, String>,
    pub metrics: Vec<Metric>,
}

impl Scrape {
    pub fn from_bytes(d: &[u8]) -> io::Result<Self> {
        log::info!("scraping {} bytes", human_readable::bytes(d.len() as f64));
        let br = BufReader::new(d);
        Self::parse(br.lines())
    }

    pub fn parse(lines: impl Iterator<Item = io::Result<String>>) -> io::Result<Self> {
        let mut docs: HashMap<String, String> = HashMap::new();
        let mut types: HashMap<String, MetricKind> = HashMap::new();
        let mut buckets: HashMap<String, Metric> = HashMap::new();
        let mut metrics: Vec<Metric> = vec![];
        for line in lines {
            let cur = match line {
                Ok(v) => v,
                Err(e) => return Err(e),
            };

            match Entry::parse_line(&cur) {
                Entry::Doc {
                    ref metric_name,
                    ref doc,
                } => {
                    docs.insert(metric_name.to_string(), doc.to_string());
                }
                Entry::Type {
                    ref metric_name,
                    ref sample_type,
                } => {
                    types.insert(metric_name.to_string(), *sample_type);
                }
                Entry::Metric {
                    metric_name,
                    ref labels,
                    value,
                    timestamp,
                } => {
                    // Parse value or skip
                    let fvalue = if let Ok(v) = parse_golang_float(value) {
                        v
                    } else {
                        continue;
                    };
                    // Parse timestamp or use given sample time
                    let timestamp = if let Some(Ok(ts_millis)) = timestamp.map(|x| x.parse::<i64>())
                    {
                        Some(Utc.timestamp_millis(ts_millis))
                    } else {
                        None
                    };
                    match (types.get(metric_name), labels) {
                        (Some(MetricKind::Histogram), Some(labels)) => {
                            if let Some(lt) = parse_bucket(labels, "le") {
                                let sample =
                                    buckets.entry(metric_name.to_string()).or_insert(Metric {
                                        metric: metric_name.to_string(),
                                        labels: None,
                                        value: Value::Histogram(vec![]),
                                        timestamp,
                                    });
                                sample.value.push_histogram(HistogramCount {
                                    less_than: lt,
                                    count: fvalue,
                                })
                            }
                        }
                        (Some(MetricKind::Summary), Some(labels)) => {
                            if let Some(q) = parse_bucket(labels, "quantile") {
                                let sample =
                                    buckets.entry(metric_name.to_string()).or_insert(Metric {
                                        metric: metric_name.to_string(),
                                        labels: None,
                                        value: Value::Summary(vec![]),
                                        timestamp,
                                    });
                                sample.value.push_summary(SummaryCount {
                                    quantile: q,
                                    count: fvalue,
                                })
                            }
                        }
                        (ty, labels) => {
                            let labels = {
                                if labels.is_some() {
                                    Some(Labels::from_str(labels.unwrap()))
                                } else {
                                    None
                                }
                            };
                            metrics.push(Metric {
                                metric: metric_name.to_string(),
                                labels,
                                value: match ty {
                                    Some(MetricKind::Counter) => Value::Counter(fvalue),
                                    Some(MetricKind::Gauge) => Value::Gauge(fvalue),
                                    _ => Value::Untyped(fvalue),
                                },
                                timestamp,
                            });
                        }
                    };
                }
                _ => {}
            }
        }
        metrics.extend(buckets.drain().map(|(_k, v)| v).collect::<Vec<_>>());
        Ok(Scrape { docs, metrics })
    }
}

lazy_static! {
    static ref NOT_FOUND_METRIC: Metric = Metric {
        metric: "not_found".to_string(),
        value: Value::Gauge(0.0),
        ..Default::default()
    };
}

/// Returns the first metric that evaluates to "true" for the function.
pub fn find_first<'a, F>(data: &'a [Metric], f: F) -> &'a Metric
where
    for<'r> F: FnMut(&'r &'a Metric) -> bool,
{
    let metric = data.iter().find(f);
    if let Some(v) = metric {
        return v;
    }
    &NOT_FOUND_METRIC
}

/// RUST_LOG=debug cargo test --all-features --package prometheus-manager --lib -- test_find_first --exact --show-output
#[test]
fn test_find_first() {
    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .is_test(true)
        .try_init();

    let scrape = r#"
# HELP http_requests_total The total number of HTTP requests.
# TYPE http_requests_total counter
http_requests_total{method="post",code="200"} 1027 1395066363000
http_requests_total{method="post",code="400"}    3 1395066363000

# Escaping in label values:
msdos_file_access_time_seconds{path="C:\\DIR\\FILE.TXT",error="Cannot find file:\n\"FILE.TXT\""} 1.458255915e9

# Minimalistic line:
metric_without_timestamp_and_labels 12.47

# TYPE metric_counter_with_no_label counter
metric_counter_with_no_label 100.10

# A weird metric from before the epoch:
something_weird{problem="division by zero"} +Inf -3982045

# A histogram, which has a pretty complex representation in the text format:
# HELP http_request_duration_seconds A histogram of the request duration.
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.05"} 24054
http_request_duration_seconds_bucket{le="0.1"} 33444
http_request_duration_seconds_bucket{le="0.2"} 100392
http_request_duration_seconds_bucket{le="0.5"} 129389
http_request_duration_seconds_bucket{le="1"} 133988
http_request_duration_seconds_bucket{le="+Inf"} 144320
http_request_duration_seconds_sum 53423
http_request_duration_seconds_count 144320

# Finally a summary, which has a complex representation, too:
# HELP rpc_duration_seconds A summary of the RPC duration in seconds.
# TYPE rpc_duration_seconds summary
rpc_duration_seconds{quantile="0.01"} 3102
rpc_duration_seconds{quantile="0.05"} 3272
rpc_duration_seconds{quantile="0.5"} 4773
rpc_duration_seconds{quantile="0.9"} 9001
rpc_duration_seconds{quantile="0.99"} 76656
rpc_duration_seconds_sum 1.7560473e+07
rpc_duration_seconds_count 2693

# TYPE avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_has_size_sum gauge
avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_has_size_sum 441892
# HELP avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_has_sum Sum of time (in ns) of a has
# TYPE avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_has_sum gauge
avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_has_sum 4.50437507e+08
"#;

    let s = Scrape::from_bytes(scrape.as_bytes()).unwrap();
    assert_eq!(s.metrics.len(), 14);

    assert_eq!(
        find_first(&s.metrics, |s| s.metric
            == "metric_without_timestamp_and_labels"),
        &Metric {
            metric: "metric_without_timestamp_and_labels".to_string(),
            value: Value::Untyped(12.47),
            ..Default::default()
        }
    );

    assert_eq!(
        find_first(&s.metrics, |s| s.metric == "metric_counter_with_no_label"),
        &Metric {
            metric: "metric_counter_with_no_label".to_string(),
            value: Value::Counter(100.10),
            ..Default::default()
        }
    );

    assert_eq!(
        find_first(&s.metrics, |s| s.metric == "http_requests_total"
            && s.labels.clone().unwrap().get("code") == Some("200")),
        &Metric {
            metric: "http_requests_total".to_string(),
            value: Value::Counter(1027f64),
            labels: Some(Labels(
                [("method", "post"), ("code", "200")]
                    .iter()
                    .map(pair_to_string)
                    .collect()
            )),
            timestamp: Some(Utc.timestamp_millis(1395066363000)),
        }
    );

    assert_eq!(
        find_first(&s.metrics, |s| s.metric == "http_requests_total"
            && s.labels.clone().unwrap().get("code") == Some("400")),
        &Metric {
            metric: "http_requests_total".to_string(),
            value: Value::Counter(3f64),
            labels: Some(Labels(
                [("method", "post"), ("code", "400")]
                    .iter()
                    .map(pair_to_string)
                    .collect()
            )),
            timestamp: Some(Utc.timestamp_millis(1395066363000)),
        }
    );

    assert_eq!(
        find_first(&s.metrics, |s| {
            s.metric
                == "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_has_size_sum"
        }),
        &Metric {
            metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_has_size_sum"
                .to_string(),
            value: Value::Gauge(441892f64),
            ..Default::default()
        }
    );

    assert_eq!(
        find_first(&s.metrics, |s| {
            s.metric == "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_has_sum"
        }),
        &Metric {
            metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_has_sum"
                .to_string(),
            value: Value::Gauge(4.50437507e+08f64),
            ..Default::default()
        }
    );
}

/// Returns all metrics that evaluate to "true" for the function.
pub fn find_all<'a, F>(data: &'a [Metric], f: F) -> Vec<&'a Metric>
where
    for<'r> F: FnMut(&'r &'a Metric) -> bool,
{
    let mut metrics: Vec<&'a Metric> = Vec::new();

    let mut iter = data.iter().filter(f);
    loop {
        let metric = iter.next();
        if let Some(v) = metric {
            metrics.push(v);
            continue;
        }
        break;
    }

    metrics
}

/// RUST_LOG=debug cargo test --all-features --package prometheus-manager --lib -- test_find_all --exact --show-output
#[test]
fn test_find_all() {
    use rust_embed::RustEmbed;

    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .is_test(true)
        .try_init();

    #[derive(RustEmbed)]
    #[folder = "artifacts/"]
    #[prefix = "artifacts/"]
    struct Asset;

    let metrics_raw = Asset::get("artifacts/avalanchego.v1.7.17.metrics").unwrap();
    let metrics_raw = std::str::from_utf8(metrics_raw.data.as_ref()).unwrap();

    let s = Scrape::from_bytes(metrics_raw.as_bytes()).unwrap();
    assert_eq!(s.metrics.len(), 2127);

    assert_eq!(
        find_all(&s.metrics, |s| s.metric.contains(
            "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_batch_put_size_",
        )),
        vec![
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_batch_put_size_count"
                    .to_string(),
                value: Value::Counter(7.469948e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_batch_put_size_sum"
                    .to_string(),
                value: Value::Gauge(2.1210190212e+10f64),
                ..Default::default()
            },
        ]
    );
}

/// Returns all metrics that evaluate to "true" for the regex on the metrics name, not the label.
/// If called in a loop, use "lazy_static" to ensure that regular expressions
/// are compiled exactly once.
/// ref. https://github.com/rust-lang/regex#usage-avoid-compiling-the-same-regex-in-a-loop
pub fn match_all_by_regex(data: &[Metric], re: Regex) -> Vec<&Metric> {
    log::debug!("matching all metrics by regex {} on the name", re);
    find_all(data, |s| re.is_match(s.metric.as_str()))
}

/// RUST_LOG=debug cargo test --all-features --package prometheus-manager --lib -- test_match_all_by_regex --exact --show-output
#[test]
fn test_match_all_by_regex() {
    use rust_embed::RustEmbed;

    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .is_test(true)
        .try_init();

    #[derive(RustEmbed)]
    #[folder = "artifacts/"]
    #[prefix = "artifacts/"]
    struct Asset;

    let metrics_raw = Asset::get("artifacts/avalanchego.v1.7.17.metrics").unwrap();
    let metrics_raw = std::str::from_utf8(metrics_raw.data.as_ref()).unwrap();

    let s = Scrape::from_bytes(metrics_raw.as_bytes()).unwrap();
    assert_eq!(s.metrics.len(), 2127);

    let re = Regex::new(r"^avalanche_(([0-9a-zA-Z]+)+){3,}_db_batch_put_size[\s\S]*$").unwrap();
    assert_eq!(
        match_all_by_regex(&s.metrics, re),
        vec![
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_batch_put_size_count"
                    .to_string(),
                value: Value::Counter(7.469948e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_batch_put_size_sum"
                    .to_string(),
                value: Value::Gauge(2.1210190212e+10f64),
                ..Default::default()
            },
        ]
    );
}

/// Returns all metrics that evaluate to "true" for the regex set on the metrics name, not the label.
/// If called in a loop, use "lazy_static" to ensure that regular expressions
/// are compiled exactly once.
/// ref. https://github.com/rust-lang/regex#usage-avoid-compiling-the-same-regex-in-a-loop
pub fn match_all_by_regex_set(data: &[Metric], rset: RegexSet) -> Vec<&Metric> {
    log::debug!("matching all metrics by regex set {:?} on the name", rset);
    find_all(data, |s| rset.is_match(s.metric.as_str()))
}

/// RUST_LOG=debug cargo test --all-features --package prometheus-manager --lib -- test_match_all_by_regex_set --exact --show-output
#[test]
fn test_match_all_by_regex_set() {
    use rust_embed::RustEmbed;

    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .is_test(true)
        .try_init();

    #[derive(RustEmbed)]
    #[folder = "artifacts/"]
    #[prefix = "artifacts/"]
    struct Asset;

    let metrics_raw = Asset::get("artifacts/avalanchego.v1.7.17.metrics").unwrap();
    let metrics_raw = std::str::from_utf8(metrics_raw.data.as_ref()).unwrap();

    let s = Scrape::from_bytes(metrics_raw.as_bytes()).unwrap();
    assert_eq!(s.metrics.len(), 2127);

    lazy_static! {
        static ref REGEXES: Vec<String> = vec![
            r"^avalanche_(([0-9a-zA-Z]+)+){40,}_blks_accepted[\s\S]*$".to_string(),
            r"^avalanche_(([0-9a-zA-Z]+)+){40,}_blks_built$".to_string(),
            r"^avalanche_(([0-9a-zA-Z]+)+){40,}_blks_rejected[\s\S]*$".to_string(),
            r"^avalanche_(([0-9a-zA-Z]+)+){40,}_db_batch_put_count$".to_string(),
            r"^avalanche_(([0-9a-zA-Z]+)+){40,}_db_batch_put_sum$".to_string(),
            r"^avalanche_(([0-9a-zA-Z]+)+){40,}_last_accepted_height$".to_string(),
            r"^avalanche_(([0-9a-zA-Z]+)+){40,}_vm_eth_rpc_failure$".to_string(),
            r"^avalanche_(([0-9a-zA-Z]+)+){40,}_vm_eth_rpc_requests$".to_string(),
            r"^avalanche_(([0-9a-zA-Z]+)+){40,}_vm_eth_rpc_success$".to_string(),
            r"^avalanche_[C|P|X]_benchlist_benched_num$".to_string(),
            r"^avalanche_[C|P]_blks_accepted[\s\S]*$".to_string(),
            r"^avalanche_[C|P]_blks_accepted[\s\S]*$".to_string(),
            r"^avalanche_[C|P|X]_db_get_count$".to_string(),
            r"^avalanche_[C|P|X]_db_read_size_sum$".to_string(),
            r"^avalanche_[C|P|X]_db_write_size_sum$".to_string(),
            r"^avalanche_[C|P|X]_polls_[\s\S]*$".to_string(),
        ];
    }

    let rset = RegexSet::new(REGEXES.to_vec()).unwrap();
    assert_eq!(
        match_all_by_regex_set(&s.metrics, rset),
        vec![
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_blks_accepted_count"
                    .to_string(),
                value: Value::Counter(43240f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_blks_accepted_sum"
                    .to_string(),
                value: Value::Gauge(9.81317938649e+11f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_blks_built"
                    .to_string(),
                value: Value::Counter(4205f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_blks_rejected_count"
                    .to_string(),
                value: Value::Counter(3f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_blks_rejected_sum"
                    .to_string(),
                value: Value::Gauge(3.8554338e+07f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_batch_put_count"
                    .to_string(),
                value: Value::Counter(7.469948e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_batch_put_sum"
                    .to_string(),
                value: Value::Gauge(5.357270217e+09f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_last_accepted_height"
                    .to_string(),
                value: Value::Gauge(43240f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_eth_rpc_failure"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_eth_rpc_requests"
                    .to_string(),
                value: Value::Gauge(4.307051e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_eth_rpc_success"
                    .to_string(),
                value: Value::Gauge(4.307051e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_benchlist_benched_num"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_blks_accepted_count"
                    .to_string(),
                value: Value::Counter(27f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_blks_accepted_sum"
                    .to_string(),
                value: Value::Gauge(1.62497901e+08f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_db_get_count"
                    .to_string(),
                value: Value::Counter(37607f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_db_read_size_sum"
                    .to_string(),
                value: Value::Gauge(2.689728e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_db_write_size_sum"
                    .to_string(),
                value: Value::Gauge(396119f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_polls_failed"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_polls_successful"
                    .to_string(),
                value: Value::Counter(420f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_benchlist_benched_num"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_blks_accepted_count"
                    .to_string(),
                value: Value::Counter(39f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_blks_accepted_sum"
                    .to_string(),
                value: Value::Gauge(4.30488494e+08f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_db_get_count"
                    .to_string(),
                value: Value::Counter(212f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_db_read_size_sum"
                    .to_string(),
                value: Value::Gauge(30726f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_db_write_size_sum"
                    .to_string(),
                value: Value::Gauge(78865f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_polls_failed"
                    .to_string(),
                value: Value::Counter(7f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_polls_successful"
                    .to_string(),
                value: Value::Counter(403f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_benchlist_benched_num"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_db_get_count"
                    .to_string(),
                value: Value::Counter(7f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_db_read_size_sum"
                    .to_string(),
                value: Value::Gauge(507f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_db_write_size_sum"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_polls_failed"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_polls_successful"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
        ]
    );
}

/// Represents the metric filter rule.
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Clone)]
#[serde(rename_all = "snake_case")]
pub struct Filter {
    pub regex: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<HashMap<String, String>>,
}

/// Returns all metrics that evaluate to "true" based on the rule.
/// If no filter has a label specified, it uses RegexSet for all regexes.
/// TODO: optimize using more RegexSet without labels...
pub fn match_all_by_rules(data: &[Metric], rules: Rules) -> io::Result<Vec<&Metric>> {
    // compile regexes in advance
    // so we don't compile multiple times for each iteration
    let mut regexes: Vec<Regex> = Vec::with_capacity(rules.filters.len());
    let mut labels_exist = false;
    for r in rules.filters.iter() {
        let regex = Regex::new(r.regex.as_str()).map_err(|e| {
            Error::new(
                ErrorKind::Other,
                format!("failed to create regex {} ({})", r.regex, e),
            )
        })?;
        regexes.push(regex);

        if r.labels.is_some() {
            labels_exist = true;
        }
    }
    log::debug!(
        "matching all metrics by {} rules (labels exist {})",
        rules.filters.len(),
        labels_exist
    );

    let found = if labels_exist {
        find_all(data, |s| {
            // iterate every rule in sequence until it finds something that matches
            for (idx, r) in rules.filters.iter().enumerate() {
                let regex = &regexes[idx];

                let regex_matched = regex.is_match(&s.metric);
                if !regex_matched {
                    // regex does not match, so no need to check further with labels
                    // retry with next rule
                    continue;
                }

                let matched = if let Some(label_rules) = &r.labels {
                    // regex matches but labels exist, so need more label checks
                    let mut label_rules_matched = true;

                    // check against the incoming metric labels
                    if let Some(current_labels) = &s.labels {
                        for (k, v) in label_rules.iter() {
                            if let Some(found_value) = current_labels.get(k) {
                                if !found_value.eq(v) {
                                    // label matches are exact
                                    // even one missing label evaluates the whole check to be false
                                    label_rules_matched = false;
                                    break;
                                }
                                continue;
                            }

                            // expected label is not found in this metric, so no match
                            // no need to check other labels
                            label_rules_matched = false;
                            break;
                        }
                    } else {
                        // the current metric has no label
                        label_rules_matched = false;
                    }

                    label_rules_matched
                } else {
                    // regex matches and this rule does not have any label
                    true
                };
                if matched {
                    // this rule matches, so no need to continue on the next rule
                    return true;
                }
            }
            false
        })
    } else {
        let regexes: Vec<String> = rules.filters.iter().map(|f| f.regex.clone()).collect();
        let reset = RegexSet::new(regexes).map_err(|e| {
            Error::new(
                ErrorKind::Other,
                format!("failed to create regex set {:?} ({})", rules, e),
            )
        })?;
        match_all_by_regex_set(data, reset)
    };
    Ok(found)
}

/// Represents metrics matching rules.
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Clone)]
#[serde(rename_all = "snake_case")]
pub struct Rules {
    /// The matching is "OR", not "AND".
    /// TODO: support "AND" rule?
    pub filters: Vec<Filter>,
    // TODO: support simple maths (e.g., divide/multiply)
}

impl Rules {
    /// Loads the "Rules" from the file.
    pub fn load(file_path: &str) -> io::Result<Rules> {
        log::info!("loading Rules from {}", file_path);

        if !Path::new(file_path).exists() {
            return Err(Error::new(
                ErrorKind::NotFound,
                format!("file {} does not exists", file_path),
            ));
        }

        let f = File::open(&file_path).map_err(|e| {
            Error::new(
                ErrorKind::Other,
                format!("failed to open {} ({})", file_path, e),
            )
        })?;
        serde_yaml::from_reader(f)
            .map_err(|e| Error::new(ErrorKind::InvalidInput, format!("invalid YAML: {}", e)))
    }

    /// Syncs the "Rules" to the file.
    pub fn sync(&self, file_path: &str) -> io::Result<()> {
        log::info!("syncing Rules to '{}'", file_path);
        let path = Path::new(file_path);
        let parent_dir = path.parent().expect("unexpected None file path parent");
        fs::create_dir_all(parent_dir)?;

        let ret = serde_yaml::to_string(self);
        let d = match ret {
            Ok(d) => d,
            Err(e) => {
                return Err(Error::new(
                    ErrorKind::Other,
                    format!("failed to serialize Node to YAML {}", e),
                ));
            }
        };
        let mut f = File::create(file_path)?;
        f.write_all(d.as_bytes())?;

        Ok(())
    }
}

/// RUST_LOG=debug cargo test --all-features --package prometheus-manager --lib -- test_match_all_by_rules --exact --show-output
#[test]
fn test_match_all_by_rules() {
    use rust_embed::RustEmbed;

    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .is_test(true)
        .try_init();

    #[derive(RustEmbed)]
    #[folder = "artifacts/"]
    #[prefix = "artifacts/"]
    struct Asset;

    let metrics_raw = Asset::get("artifacts/avalanchego.v1.7.17.metrics").unwrap();
    let metrics_raw = std::str::from_utf8(metrics_raw.data.as_ref()).unwrap();

    let s = Scrape::from_bytes(metrics_raw.as_bytes()).unwrap();
    assert_eq!(s.metrics.len(), 2127);

    let rules = Rules::load("artifacts/avalanchego.rules.yaml").unwrap();
    assert_eq!(
        match_all_by_rules(&s.metrics, rules).unwrap(),
        vec![
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_blks_accepted_count"
                    .to_string(),
                value: Value::Counter(43240f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_blks_accepted_sum"
                    .to_string(),
                value: Value::Gauge(9.81317938649e+11f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_blks_built"
                    .to_string(),
                value: Value::Counter(4205f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_blks_processing"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_blks_rejected_count"
                    .to_string(),
                value: Value::Counter(3f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_blks_rejected_sum"
                    .to_string(),
                value: Value::Gauge(3.8554338e+07f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_batch_put_count"
                    .to_string(),
                value: Value::Counter(7.469948e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_batch_put_sum"
                    .to_string(),
                value: Value::Gauge(5.357270217e+09f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_batch_write_count"
                    .to_string(),
                value: Value::Counter(259476f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_batch_write_sum"
                    .to_string(),
                value: Value::Gauge(2.8494818524e+10f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_compact_count"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_compact_sum"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_delete_count"
                    .to_string(),
                value: Value::Counter(4f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_delete_sum"
                    .to_string(),
                value: Value::Gauge(66771f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_get_count"
                    .to_string(),
                value: Value::Counter(5.703759e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_get_sum"
                    .to_string(),
                value: Value::Gauge(3.5932638264e+11f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_put_count"
                    .to_string(),
                value: Value::Counter(1309f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_db_put_sum"
                    .to_string(),
                value: Value::Gauge(5.6261023e+07f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_handler_get_accepted_count"
                    .to_string(),
                value: Value::Counter(10f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_handler_get_accepted_sum"
                    .to_string(),
                value: Value::Gauge(132034f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_last_accepted_height"
                    .to_string(),
                value: Value::Gauge(43240f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_polls_successful"
                    .to_string(),
                value: Value::Counter(649334f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_eth_rpc_failure"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_eth_rpc_requests"
                    .to_string(),
                value: Value::Gauge(4.307051e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_eth_rpc_success"
                    .to_string(),
                value: Value::Gauge(4.307051e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_grpc_client_handled_total".to_string(),
                labels: Some(Labels(
                    [("grpc_code", "OK"), ("grpc_method", "Delete"), ("grpc_service", "rpcdb.Database"), ("grpc_type", "unary")]
                        .iter()
                        .map(pair_to_string)
                        .collect()
                )),
                value: Value::Counter(4f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_grpc_client_handled_total".to_string(),
                labels: Some(Labels(
                    [("grpc_code", "OK"), ("grpc_method", "Get"), ("grpc_service", "rpcdb.Database"), ("grpc_type", "unary")]
                        .iter()
                        .map(pair_to_string)
                        .collect()
                )),
                value: Value::Counter(5.578128e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_grpc_client_handled_total".to_string(),
                labels: Some(Labels(
                    [("grpc_code", "OK"), ("grpc_method", "Has"), ("grpc_service", "rpcdb.Database"), ("grpc_type", "unary")]
                        .iter()
                        .map(pair_to_string)
                        .collect()
                )),
                value: Value::Counter(4209f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_grpc_client_handled_total".to_string(),
                labels: Some(Labels(
                    [("grpc_code", "OK"), ("grpc_method", "HealthCheck"), ("grpc_service", "rpcdb.Database"), ("grpc_type", "unary")]
                        .iter()
                        .map(pair_to_string)
                        .collect()
                )),
                value: Value::Counter(12879f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_grpc_client_handled_total".to_string(),
                labels: Some(Labels(
                    [("grpc_code", "OK"), ("grpc_method", "IteratorNext"), ("grpc_service", "rpcdb.Database"), ("grpc_type", "unary")]
                        .iter()
                        .map(pair_to_string)
                        .collect()
                )),
                value: Value::Counter(3f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_grpc_client_handled_total".to_string(),
                labels: Some(Labels(
                    [("grpc_code", "OK"), ("grpc_method", "IteratorRelease"), ("grpc_service", "rpcdb.Database"), ("grpc_type", "unary")]
                        .iter()
                        .map(pair_to_string)
                        .collect()
                )),
                value: Value::Counter(2f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_grpc_client_handled_total".to_string(),
                labels: Some(Labels(
                    [("grpc_code", "OK"), ("grpc_method", "NewIteratorWithStartAndPrefix"), ("grpc_service", "rpcdb.Database"), ("grpc_type", "unary")]
                        .iter()
                        .map(pair_to_string)
                        .collect()
                )),
                value: Value::Counter(2f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_grpc_client_handled_total".to_string(),
                labels: Some(Labels(
                    [("grpc_code", "OK"), ("grpc_method", "Put"), ("grpc_service", "rpcdb.Database"), ("grpc_type", "unary")]
                        .iter()
                        .map(pair_to_string)
                        .collect()
                )),
                value: Value::Counter(1309f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_grpc_client_handled_total".to_string(),
                labels: Some(Labels(
                    [("grpc_code", "OK"), ("grpc_method", "SendAppGossip"), ("grpc_service", "appsender.AppSender"), ("grpc_type", "unary")]
                        .iter()
                        .map(pair_to_string)
                        .collect()
                )),
                value: Value::Counter(257127f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_grpc_client_handled_total".to_string(),
                labels: Some(Labels(
                    [("grpc_code", "OK"), ("grpc_method", "WriteBatch"), ("grpc_service", "rpcdb.Database"), ("grpc_type", "unary")]
                        .iter()
                        .map(pair_to_string)
                        .collect()
                )),
                value: Value::Counter(259475f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_grpc_client_started_total".to_string(),
                labels: Some(Labels(
                    [("grpc_method", "Delete"), ("grpc_service", "rpcdb.Database"), ("grpc_type", "unary")]
                        .iter()
                        .map(pair_to_string)
                        .collect()
                )),
                value: Value::Counter(4f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_grpc_client_started_total".to_string(),
                labels: Some(Labels(
                    [("grpc_method", "Get"), ("grpc_service", "rpcdb.Database"), ("grpc_type", "unary")]
                        .iter()
                        .map(pair_to_string)
                        .collect()
                )),
                value: Value::Counter(5.578128e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_grpc_client_started_total".to_string(),
                labels: Some(Labels(
                    [("grpc_method", "Has"), ("grpc_service", "rpcdb.Database"), ("grpc_type", "unary")]
                        .iter()
                        .map(pair_to_string)
                        .collect()
                )),
                value: Value::Counter(4209f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_grpc_client_started_total".to_string(),
                labels: Some(Labels(
                    [("grpc_method", "HealthCheck"), ("grpc_service", "rpcdb.Database"), ("grpc_type", "unary")]
                        .iter()
                        .map(pair_to_string)
                        .collect()
                )),
                value: Value::Counter(12879f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_grpc_client_started_total".to_string(),
                labels: Some(Labels(
                    [("grpc_method", "IteratorNext"), ("grpc_service", "rpcdb.Database"), ("grpc_type", "unary")]
                        .iter()
                        .map(pair_to_string)
                        .collect()
                )),
                value: Value::Counter(3f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_grpc_client_started_total".to_string(),
                labels: Some(Labels(
                    [("grpc_method", "IteratorRelease"), ("grpc_service", "rpcdb.Database"), ("grpc_type", "unary")]
                        .iter()
                        .map(pair_to_string)
                        .collect()
                )),
                value: Value::Counter(2f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_grpc_client_started_total".to_string(),
                labels: Some(Labels(
                    [("grpc_method", "NewIteratorWithStartAndPrefix"), ("grpc_service", "rpcdb.Database"), ("grpc_type", "unary")]
                        .iter()
                        .map(pair_to_string)
                        .collect()
                )),
                value: Value::Counter(2f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_grpc_client_started_total".to_string(),
                labels: Some(Labels(
                    [("grpc_method", "Put"), ("grpc_service", "rpcdb.Database"), ("grpc_type", "unary")]
                        .iter()
                        .map(pair_to_string)
                        .collect()
                )),
                value: Value::Counter(1309f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_grpc_client_started_total".to_string(),
                labels: Some(Labels(
                    [("grpc_method", "SendAppGossip"), ("grpc_service", "appsender.AppSender"), ("grpc_type", "unary")]
                        .iter()
                        .map(pair_to_string)
                        .collect()
                )),
                value: Value::Counter(257127f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_grpc_client_started_total".to_string(),
                labels: Some(Labels(
                    [("grpc_method", "WriteBatch"), ("grpc_service", "rpcdb.Database"), ("grpc_type", "unary")]
                        .iter()
                        .map(pair_to_string)
                        .collect()
                )),
                value: Value::Counter(259475f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_metervm_parse_block_count"
                    .to_string(),
                value: Value::Counter(1.121427e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_metervm_parse_block_sum"
                    .to_string(),
                value: Value::Gauge(5.82743213583e+11f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_benchlist_benched_num"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_blks_accepted_count"
                    .to_string(),
                value: Value::Counter(27f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_blks_accepted_sum"
                    .to_string(),
                value: Value::Gauge(1.62497901e+08f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_blks_built"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_blks_processing"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_blks_rejected_count"
                    .to_string(),
                value: Value::Counter(3f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_blks_rejected_sum"
                    .to_string(),
                value: Value::Gauge(1.8306179e+07f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_db_batch_put_count"
                    .to_string(),
                value: Value::Counter(659f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_db_batch_put_sum"
                    .to_string(),
                value: Value::Gauge(450756f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_db_batch_write_count"
                    .to_string(),
                value: Value::Counter(176f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_db_batch_write_sum"
                    .to_string(),
                value: Value::Gauge(2.894071e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_db_compact_count"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_db_compact_sum"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_db_delete_count"
                    .to_string(),
                value: Value::Counter(2f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_db_delete_sum"
                    .to_string(),
                value: Value::Gauge(11070f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_db_get_count"
                    .to_string(),
                value: Value::Counter(37607f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_db_get_sum"
                    .to_string(),
                value: Value::Gauge(1.22077894e+09f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_db_put_count"
                    .to_string(),
                value: Value::Counter(1290f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_db_put_sum"
                    .to_string(),
                value: Value::Gauge(3.2048842e+07f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_polls_failed"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_polls_successful"
                    .to_string(),
                value: Value::Counter(420f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_vm_metervm_parse_block_count"
                    .to_string(),
                value: Value::Counter(348187f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_C_vm_metervm_parse_block_sum"
                    .to_string(),
                value: Value::Gauge(2.364194114e+09f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_benchlist_benched_num"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_blks_accepted_count"
                    .to_string(),
                value: Value::Counter(39f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_blks_accepted_sum"
                    .to_string(),
                value: Value::Gauge(4.30488494e+08f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_blks_built"
                    .to_string(),
                value: Value::Counter(3f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_blks_processing"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_blks_rejected_count"
                    .to_string(),
                value: Value::Counter(21f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_blks_rejected_sum"
                    .to_string(),
                value: Value::Gauge(5.25170283e+08f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_db_batch_put_count"
                    .to_string(),
                value: Value::Counter(322f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_db_batch_put_sum"
                    .to_string(),
                value: Value::Gauge(116272f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_db_batch_write_count"
                    .to_string(),
                value: Value::Counter(95f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_db_batch_write_sum"
                    .to_string(),
                value: Value::Gauge(1.371173e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_db_compact_count"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_db_compact_sum"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_db_delete_count"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_db_delete_sum"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_db_get_count"
                    .to_string(),
                value: Value::Counter(212f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_db_get_sum"
                    .to_string(),
                value: Value::Gauge(1.139713e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_db_put_count"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_db_put_sum"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_polls_failed"
                    .to_string(),
                value: Value::Counter(7f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_polls_successful"
                    .to_string(),
                value: Value::Counter(403f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_vm_metervm_parse_block_count"
                    .to_string(),
                value: Value::Counter(348418f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_P_vm_metervm_parse_block_sum"
                    .to_string(),
                value: Value::Gauge(5.5101287211e+10f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_benchlist_benched_num"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_db_batch_put_count"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_db_batch_put_sum"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_db_batch_write_count"
                    .to_string(),
                value: Value::Counter(4f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_db_batch_write_sum"
                    .to_string(),
                value: Value::Gauge(2070f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_db_compact_count"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_db_compact_sum"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_db_delete_count"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_db_delete_sum"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_db_get_count"
                    .to_string(),
                value: Value::Counter(7f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_db_get_sum"
                    .to_string(),
                value: Value::Gauge(38260f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_db_put_count"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_db_put_sum"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_polls_failed"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_polls_successful"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_whitelist_tx_accepted_count"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_whitelist_tx_accepted_sum"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_whitelist_tx_polls_accepted_count"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_whitelist_tx_polls_accepted_sum"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_whitelist_tx_polls_rejected_count"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_whitelist_tx_polls_rejected_sum"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_whitelist_tx_processing"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_whitelist_tx_rejected_count"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_whitelist_tx_rejected_sum"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_whitelist_vtx_issue_failure"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_X_whitelist_vtx_issue_success"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_db_batch_put_count"
                    .to_string(),
                value: Value::Counter(7.470929e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_db_batch_put_sum"
                    .to_string(),
                value: Value::Gauge(4.234592257e+09f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_db_batch_write_count"
                    .to_string(),
                value: Value::Counter(259751f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_db_batch_write_sum"
                    .to_string(),
                value: Value::Gauge(2.8387298672e+10f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_db_compact_count"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_db_compact_sum"
                    .to_string(),
                value: Value::Gauge(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_db_delete_count"
                    .to_string(),
                value: Value::Counter(6f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_db_delete_sum"
                    .to_string(),
                value: Value::Gauge(75452f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_db_get_count"
                    .to_string(),
                value: Value::Counter(5.741585e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_db_get_sum"
                    .to_string(),
                value: Value::Gauge(3.57862202207e+11f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_db_put_count"
                    .to_string(),
                value: Value::Counter(2599f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_db_put_sum"
                    .to_string(),
                value: Value::Gauge(8.6561039e+07f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_inbound_conn_throttler_allowed"
                    .to_string(),
                value: Value::Counter(19f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_inbound_conn_throttler_rate_limited"
                    .to_string(),
                value: Value::Counter(11f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_node_uptime_rewarding_stake"
                    .to_string(),
                value: Value::Gauge(100f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_node_uptime_weighted_average"
                    .to_string(),
                value: Value::Gauge(99.19999000049998f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_peerlist_failed"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_peerlist_received"
                    .to_string(),
                value: Value::Counter(57968f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_peerlist_sent"
                    .to_string(),
                value: Value::Counter(57975f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_peers"
                    .to_string(),
                value: Value::Gauge(9f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_peers_subnet"
                    .to_string(),
                labels: Some(Labels(
                        [("subnetID", "2At5uFe2kDiYsHziSqJeebizvF9zQbH4m9mPQbKdMKhEWJj5AW")]
                            .iter()
                            .map(pair_to_string)
                            .collect()
                    )),
                value: Value::Gauge(9f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_ping_failed"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_ping_received"
                    .to_string(),
                value: Value::Counter(154521f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_ping_sent"
                    .to_string(),
                value: Value::Counter(154530f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_pong_failed"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_pong_received"
                    .to_string(),
                value: Value::Counter(154521f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_pong_sent"
                    .to_string(),
                value: Value::Counter(154521f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_pull_query_failed"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_pull_query_received"
                    .to_string(),
                value: Value::Counter(5.740569e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_pull_query_sent"
                    .to_string(),
                value: Value::Counter(5.72831e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_push_query_failed"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_push_query_received"
                    .to_string(),
                value: Value::Counter(342488f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_push_query_sent"
                    .to_string(),
                value: Value::Counter(342523f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_put_failed"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_put_received"
                    .to_string(),
                value: Value::Counter(1.437452e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_put_sent"
                    .to_string(),
                value: Value::Counter(1.437383e+06f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_throttler_outbound_acquire_failures"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_throttler_outbound_acquire_successes"
                    .to_string(),
                value: Value::Counter(1.627736e+07f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_times_connected"
                    .to_string(),
                value: Value::Counter(24f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_times_disconnected"
                    .to_string(),
                value: Value::Counter(15f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_version_failed"
                    .to_string(),
                value: Value::Counter(0f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_version_received"
                    .to_string(),
                value: Value::Counter(24f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_network_version_sent"
                    .to_string(),
                value: Value::Counter(24f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_process_max_fds"
                    .to_string(),
                value: Value::Gauge(32768f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_process_open_fds"
                    .to_string(),
                value: Value::Gauge(463f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_process_resident_memory_bytes"
                    .to_string(),
                value: Value::Gauge(3.33867008e+09f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_process_virtual_memory_bytes"
                    .to_string(),
                value: Value::Gauge(6.609301504e+09f64),
                ..Default::default()
            },
            &Metric {
                metric: "avalanche_7y7zwo7XatqnX4dtTakLo32o7jkMX4XuDa26WaxbCXoCT1qKK_vm_eth_rpc_duration_all"
                    .to_string(),
                value: Value::Summary(vec![
                    SummaryCount{quantile: 0.5, count: 327057.0f64},
                    SummaryCount{quantile: 0.75, count: 382043.0f64},
                    SummaryCount{quantile: 0.95, count: 2659645.649999993f64},
                    SummaryCount{quantile: 0.99, count: 198345217.87000003f64},
                    SummaryCount{quantile: 0.999, count: 200404035.783f64},
                    SummaryCount{quantile: 0.9999, count: 200408762.0f64},
                    ]),
                ..Default::default()
            },
        ]
    );
}

pub fn pair_to_string(pair: &(&str, &str)) -> (String, String) {
    (pair.0.to_string(), pair.1.to_string())
}

fn parse_golang_float(s: &str) -> Result<f64, <f64 as std::str::FromStr>::Err> {
    match s.to_lowercase().as_str() {
        "nan" => Ok(std::f64::NAN), // f64::parse doesn't recognize 'nan'
        s => s.parse::<f64>(),      // f64::parse expects lowercase [+-]inf
    }
}

/// RUST_LOG=debug cargo test --all-features --package prometheus-manager --lib -- test_golang_float --exact --show-output
#[test]
fn test_golang_float() {
    assert_eq!(parse_golang_float("1.0"), Ok(1.0f64));
    assert_eq!(parse_golang_float("-1.0"), Ok(-1.0f64));
    assert!(parse_golang_float("NaN").unwrap().is_nan());
    assert_eq!(parse_golang_float("Inf"), Ok(std::f64::INFINITY));
    assert_eq!(parse_golang_float("+Inf"), Ok(std::f64::INFINITY));
    assert_eq!(parse_golang_float("-Inf"), Ok(std::f64::NEG_INFINITY));
}
