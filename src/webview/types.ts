// Shared type definitions for the webview data flow

export interface TrainingData {
    accuracy: number;
    protected_features: string[];
    num_parameters: number;
    hidden_layers?: number[];
    training_history?: {
        final_train_loss: number;
        final_val_loss: number;
        final_train_acc: number;
        final_val_acc: number;
        epochs_trained: number;
    };
    dataset_info?: {
        num_features: number;
        num_train: number;
        num_val: number;
        num_test: number;
        num_total: number;
        class_distribution: Record<string, number>;
        feature_names: string[];
        protected_attr_info: Record<string, { index: number; num_unique_values: number }>;
    };
}

export interface QidMetrics {
    mean_qid: number;
    max_qid: number;
    pct_discriminatory: number;
    num_discriminatory: number;
    mean_disparate_impact: number;
    num_analyzed: number;
}

export interface SearchResults {
    discriminatory_instances: { qid: number; variance: number }[];
    best_qid: number;
    num_found: number;
}

export interface LayerAnalysis {
    biased_layer: { sensitivity: number; layer_name: string; neuron_count: number; layer_idx: number };
    all_layers: { layer_name: string; sensitivity: number }[];
}

export interface NeuronAnalysis {
    neuron_idx: number;
    impact_score: number;
}

export interface ShapData {
    shap_values: number[][];
    global_importance: number[];
    feature_names: string[];
    base_value: number;
    num_explained: number;
    feature_values: number[][];
}

export interface LimeData {
    aggregated_importance: number[];
    feature_names: string[];
    num_explained: number;
    explanations: {
        instance_idx: number;
        feature_weights: [string, number][];
        prediction_proba: number[];
    }[];
}

export interface ActivationsData {
    method: string;
    num_samples: number;
    layers: { layer_name: string; x: number[]; y: number[] }[];
    labels: number[];
    protected: number[];
}

export interface AnalysisMetadata {
    file?: string;
    filePath?: string;
    labelColumn?: string;
    totalTime?: number;
    protectedFeatures?: string[];
    hiddenLayers?: number[];
    epochs?: number;
    maxSamples?: number;
    globalIterations?: number;
    localNeighbors?: number;
    numExplainInstances?: number;
    stepTimings?: Record<string, number>;
}

export interface AnalysisResults {
    training: TrainingData;
    analysis: { qid_metrics: QidMetrics };
    search: { search_results: SearchResults };
    debug?: { layer_analysis: LayerAnalysis; neuron_analysis: NeuronAnalysis[] };
    activations?: ActivationsData;
    explanations?: { shap?: ShapData; lime?: LimeData };
    metadata?: AnalysisMetadata;
}
