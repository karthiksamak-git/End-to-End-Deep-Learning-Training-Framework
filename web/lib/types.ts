/** Types mirroring the JSON artifacts written by `dltf export`. */

export type HistoryPoint = {
  epoch: number;
  train_loss: number;
  train_accuracy: number;
  train_f1: number;
  val_loss: number;
  val_accuracy: number;
  val_f1: number;
  val_roc_auc: number | null;
  lr: number;
  seconds: number;
};

export type DeviceInfo = {
  device?: string;
  kind?: string;
  amp_dtype?: string | null;
  supports_amp?: boolean;
  notes?: string;
};

export type ModelInfo = {
  class?: string;
  parameters_total?: number;
  parameters_trainable?: number;
  size_mb_fp32?: number;
};

export type Run = {
  id: string;
  name: string;
  task: string | null;
  created_at: string | null;
  duration_sec: number | null;
  epochs_run: number | null;
  seed: number | null;
  device: DeviceInfo;
  git_hash: string | null;
  model: ModelInfo;
  config: Record<string, unknown>;
  history: HistoryPoint[];
  best: { epoch?: number; score?: number; metric?: string };
  best_record: HistoryPoint | null;
  final: HistoryPoint | null;
  test_metrics: Record<string, number>;
  sizes: Record<string, number>;
};

export type ExperimentsFile = {
  generated_at: string;
  runs: Run[];
};

export type CvFold = {
  fold: number;
  train_size: number;
  val_size: number;
  test_size: number;
  best_epoch: number;
  epochs_run: number;
  accuracy: number;
  f1_macro: number;
  precision_macro: number;
  recall_macro: number;
  roc_auc: number;
  log_loss: number;
};

export type CvFile = {
  task: string;
  dataset: string;
  model: string;
  n_splits: number;
  epochs: number;
  seed: number;
  device?: DeviceInfo;
  folds: CvFold[];
  summary: Record<string, { mean: number; std: number }>;
  generated_at: string;
};

export type HpoTrial = {
  number: number;
  state: string;
  value: number | null;
  params: Record<string, string | number>;
  duration_sec: number;
  started_at?: string | null;
};

export type HpoFile = {
  study_name: string;
  direction: string;
  metric: string;
  n_trials: number;
  n_completed: number;
  n_pruned: number;
  best_value: number | null;
  best_params: Record<string, string | number>;
  best_trial: number | null;
  param_importance: Record<string, number>;
  search_space: Record<string, Array<string | number>>;
  epochs_per_trial: number;
  subset_train: number;
  subset_val: number;
  seed: number;
  trials: HpoTrial[];
  generated_at: string;
};

export type OnnxInfo = {
  path?: string;
  size_bytes?: number;
  size_mb?: number;
  under_10mb?: boolean;
  opset?: number;
  input_shape?: number[];
  input_name?: string;
  output_name?: string;
  dynamic_batch?: boolean;
};

export type ModelCard = {
  name: string;
  generated_at: string;
  git_hash: string;
  seed: number | null;
  device: DeviceInfo;
  task: string | null;
  dataset: {
    name?: string;
    train_size?: number;
    val_size?: number;
    test_size?: number;
    image_mean?: number;
    image_std?: number;
    preprocessing?: string;
  };
  architecture: {
    class?: string;
    parameters_total?: number;
    size_mb_fp32?: number;
    config?: Record<string, unknown>;
  };
  training: {
    epochs_run?: number;
    duration_sec?: number;
    optimizer?: string;
    lr?: number;
    weight_decay?: number;
    scheduler?: string;
    batch_size?: number;
    amp?: string | null;
  };
  metrics: Record<string, number>;
  onnx: OnnxInfo;
  onnx_parity: {
    max_abs_diff?: number;
    atol?: number;
    passed?: boolean;
    inputs_checked?: number;
  };
  class_names: string[] | null;
  limitations: string[];
};

export type Headline = {
  test_accuracy: number | null;
  test_f1_macro: number | null;
  cv_accuracy_mean: number | null;
  cv_accuracy_std: number | null;
  hpo_best_value: number | null;
  hpo_metric: string | null;
  n_runs: number;
};

export type PerClassRow = {
  class: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
};

export type MisclassifiedItem = {
  sample_index: number;
  true_index: number;
  pred_index: number;
  true_label: string;
  pred_label: string;
  confidence: number;
  top3: Array<{ label: string; probability: number }>;
  pixels?: number[];
};

export type ErrorAnalysisFile = {
  task: string;
  num_samples: number;
  num_misclassified: number;
  class_names: string[];
  overall: Record<string, number | null>;
  confusion_matrix: { labels: string[]; matrix: number[][] };
  per_class: PerClassRow[];
  misclassified: { total: number; items: MisclassifiedItem[] };
  top_confusions: Array<{ true_label: string; pred_label: string; count: number }>;
  calibration: {
    ece: number;
    bins: Array<{
      bin: number;
      lower: number;
      upper: number;
      count: number;
      avg_confidence: number;
      accuracy: number;
    }>;
  };
  findings_markdown: string;
};

export type PlaygroundSample = {
  index: number;
  label: number;
  label_name: string;
  pixels: number[];
};
