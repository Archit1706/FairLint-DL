import { getStyles } from './styles';
import { ICONS } from './icons';
import { calculateFairnessScore, getFairnessStatus } from './scoring';
import { getChartsScript } from './charts';

export function getWebviewHtml(results: {
    training: { accuracy: number; protected_features: string[]; num_parameters: number };
    analysis: { qid_metrics: Record<string, number> };
    search: { search_results: { discriminatory_instances: unknown[]; best_qid: number; num_found: number } };
    debug?: { layer_analysis: Record<string, unknown>; neuron_analysis: unknown[] };
    activations?: Record<string, unknown>;
    explanations?: { shap?: Record<string, unknown>; lime?: Record<string, unknown> };
    metadata?: { file?: string; labelColumn?: string; totalTime?: number };
}): string {
    const qidMetrics = results.analysis.qid_metrics as Record<string, number>;
    const searchResults = results.search.search_results as {
        discriminatory_instances: unknown[];
        best_qid: number;
        num_found: number;
    };
    const layerAnalysis = results.debug?.layer_analysis as {
        biased_layer: { sensitivity: number; layer_name: string; neuron_count: number };
        all_layers: unknown[];
    } | null;
    const neuronAnalysis = (results.debug?.neuron_analysis as { neuron_idx: number; impact_score: number }[]) || null;
    const activationsData = results.activations || null;
    const explanationsData = (results.explanations as { shap?: Record<string, unknown>; lime?: Record<string, unknown> }) || null;
    const metadata = results.metadata;

    const fairnessScore = calculateFairnessScore(qidMetrics as {
        mean_qid: number; max_qid: number; pct_discriminatory: number; mean_disparate_impact: number;
    });
    const fairnessStatus = getFairnessStatus(fairnessScore);

    const headScript = `
    <script>
        var _vscodeApi = (function() { try { return acquireVsCodeApi(); } catch(e) { return null; } })();
        function downloadJson() {
            if (!_vscodeApi) return;
            try { _vscodeApi.postMessage({ command: 'saveJson' }); } catch (e) { console.error('Download JSON error:', e); }
        }
    </script>`;

    const chartsScript = getChartsScript(
        searchResults,
        qidMetrics as { mean_disparate_impact: number },
        layerAnalysis,
        neuronAnalysis,
        activationsData,
        explanationsData as { shap: unknown | null; lime: unknown | null } | null,
    );

    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    ${headScript}
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>${getStyles()}</style>
</head>
<body>
    ${buildHeader(metadata, fairnessScore, fairnessStatus)}
    ${buildTrainingSection(results.training)}
    ${buildActivationsSection(activationsData)}
    ${buildQidSection(qidMetrics)}
    ${buildSearchSection(searchResults)}
    ${buildLayerSection(layerAnalysis, neuronAnalysis)}
    ${buildShapSection(explanationsData)}
    ${buildLimeSection(explanationsData)}
    ${buildInterpretationSection()}
    ${chartsScript}
</body>
</html>`;
}

function buildHeader(
    metadata: { file?: string; labelColumn?: string; totalTime?: number } | undefined,
    fairnessScore: number,
    fairnessStatus: { label: string; class: string; color: string },
): string {
    return `
    <div class="header">
        <div class="header-left">
            <h1>Fairness Analysis Results</h1>
            <div class="header-meta">
                <span>${ICONS.file} ${metadata?.file || 'Dataset'}</span>
                <span>${ICONS.tag} Label: ${metadata?.labelColumn || 'N/A'}</span>
                <span>${ICONS.clock} ${metadata?.totalTime || 0}s analysis time</span>
            </div>
            <div class="export-toolbar">
                <button class="export-btn" onclick="downloadJson()" title="Download raw analysis data as JSON">
                    ${ICONS.download} Export JSON
                </button>
            </div>
        </div>
        <div class="score-card">
            <div class="score-value" style="color: ${fairnessStatus.color}">${fairnessScore}</div>
            <div class="score-label">Fairness Score</div>
            <div class="score-status ${fairnessStatus.class}">${fairnessStatus.label}</div>
        </div>
    </div>`;
}

function buildTrainingSection(training: { accuracy: number; protected_features: string[]; num_parameters: number }): string {
    return `
    <div class="section">
        <div class="section-header">
            <div class="section-icon" style="background: rgba(76, 175, 80, 0.2); color: var(--accent-green);">
                ${ICONS.barChart}
            </div>
            <h2 class="section-title">Model Training</h2>
        </div>
        <div class="cards-grid">
            <div class="card">
                <div class="card-header">
                    <span class="card-label">Model Accuracy</span>
                    <span class="card-badge badge-success">Trained</span>
                </div>
                <div class="card-value">${training.accuracy.toFixed(1)}<span class="card-unit">%</span></div>
                <div class="card-description">Test set classification accuracy</div>
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${training.accuracy}%; background: var(--accent-green);"></div>
                    </div>
                </div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-label">Protected Features</span>
                    <span class="card-badge badge-info">User-Selected</span>
                </div>
                <div class="card-value">${training.protected_features.length}</div>
                <div class="card-description">${training.protected_features.join(', ')}</div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-label">Model Size</span>
                </div>
                <div class="card-value">${(training.num_parameters / 1000).toFixed(1)}<span class="card-unit">K params</span></div>
                <div class="card-description">Total trainable parameters</div>
            </div>
        </div>
    </div>`;
}

function buildActivationsSection(activationsData: Record<string, unknown> | null): string {
    if (!activationsData) {
        return '';
    }
    const data = activationsData as { method: string; num_samples: number; layers: { layer_name: string }[] };
    return `
    <div class="section">
        <div class="section-header">
            <div class="section-icon" style="background: rgba(156, 39, 176, 0.2); color: var(--accent-purple);">
                ${ICONS.eye}
            </div>
            <h2 class="section-title">Internal Space Visualization</h2>
        </div>
        <div class="interpretation-box" style="margin-bottom: 16px;">
            <div class="interpretation-title">${ICONS.info} How to Read These Plots</div>
            <div class="interpretation-content">
                Each chart shows the <strong>${data.method.toUpperCase()}</strong> 2D projection of layer activations for <strong>${data.num_samples}</strong> test instances.
                Points are colored by <strong>prediction label</strong>. Clusters that separate by protected attribute indicate the model is learning to encode protected information at that layer.
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 16px;">
            ${data.layers
                .map(
                    (layer: { layer_name: string }, idx: number) => `
                <div class="chart-container">
                    <div class="chart-title">${layer.layer_name} Activations (${data.method.toUpperCase()})</div>
                    <div id="activation-layer-${idx}" style="height: 350px;"></div>
                </div>`,
                )
                .join('')}
        </div>
    </div>`;
}

function buildQidSection(qidMetrics: Record<string, number>): string {
    return `
    <div class="section">
        <div class="section-header">
            <div class="section-icon" style="background: rgba(79, 195, 247, 0.2); color: var(--accent-blue);">
                ${ICONS.scale}
            </div>
            <h2 class="section-title">Fairness Metrics (QID Analysis)</h2>
        </div>
        <div class="cards-grid">
            <div class="card">
                <div class="card-header">
                    <span class="card-label">Mean QID</span>
                    <span class="card-badge ${qidMetrics.mean_qid > 1.0 ? 'badge-warning' : 'badge-success'}">${qidMetrics.mean_qid > 1.0 ? 'High' : 'Low'}</span>
                </div>
                <div class="card-value">${qidMetrics.mean_qid.toFixed(4)}<span class="card-unit">bits</span></div>
                <div class="card-description">Average protected information used in decisions. Lower is better.</div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-label">Max QID</span>
                    <span class="card-badge ${qidMetrics.max_qid > 2.0 ? 'badge-danger' : 'badge-success'}">${qidMetrics.max_qid > 2.0 ? 'Critical' : 'Normal'}</span>
                </div>
                <div class="card-value">${qidMetrics.max_qid.toFixed(4)}<span class="card-unit">bits</span></div>
                <div class="card-description">Worst-case protected information leakage.</div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-label">Discriminatory Instances</span>
                    <span class="card-badge ${qidMetrics.pct_discriminatory > 10 ? 'badge-warning' : 'badge-success'}">${qidMetrics.pct_discriminatory > 10 ? 'Concerning' : 'Acceptable'}</span>
                </div>
                <div class="card-value">${qidMetrics.num_discriminatory}<span class="card-unit">(${qidMetrics.pct_discriminatory.toFixed(1)}%)</span></div>
                <div class="card-description">Instances with QID > 0.1 bits showing potential bias.</div>
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${Math.min(qidMetrics.pct_discriminatory, 100)}%; background: ${qidMetrics.pct_discriminatory > 10 ? 'var(--accent-orange)' : 'var(--accent-green)'};"></div>
                    </div>
                    <div class="progress-labels"><span>0%</span><span>10% threshold</span><span>100%</span></div>
                </div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-label">Disparate Impact Ratio</span>
                    <span class="card-badge ${qidMetrics.mean_disparate_impact < 0.8 ? 'badge-danger' : 'badge-success'}">${qidMetrics.mean_disparate_impact < 0.8 ? 'Violation' : 'Compliant'}</span>
                </div>
                <div class="card-value">${qidMetrics.mean_disparate_impact.toFixed(3)}</div>
                <div class="card-description">Ratio should be >= 0.8 for legal compliance.</div>
                <div class="compliance-box ${qidMetrics.mean_disparate_impact >= 0.8 ? 'compliance-pass' : 'compliance-fail'}">
                    <div class="compliance-header">
                        ${qidMetrics.mean_disparate_impact >= 0.8 ? '&#10003; Passes 80% Rule' : '&#10007; Violates 80% Rule'}
                    </div>
                    <div class="compliance-text">
                        ${qidMetrics.mean_disparate_impact >= 0.8
                            ? 'Model meets legal fairness thresholds for hiring/lending decisions.'
                            : 'Model may exhibit legally actionable discrimination. Consider retraining with fairness constraints.'}
                    </div>
                </div>
            </div>
        </div>
    </div>`;
}

function buildSearchSection(searchResults: { best_qid: number; num_found: number }): string {
    return `
    <div class="section">
        <div class="section-header">
            <div class="section-icon" style="background: rgba(255, 152, 0, 0.2); color: var(--accent-orange);">
                ${ICONS.search}
            </div>
            <h2 class="section-title">Discriminatory Instance Search</h2>
        </div>
        <div class="cards-grid">
            <div class="card">
                <div class="card-header"><span class="card-label">Best QID Found</span></div>
                <div class="card-value">${searchResults.best_qid.toFixed(4)}<span class="card-unit">bits</span></div>
                <div class="card-description">Maximum discrimination discovered via gradient search.</div>
            </div>
            <div class="card">
                <div class="card-header"><span class="card-label">Instances Generated</span></div>
                <div class="card-value">${searchResults.num_found}</div>
                <div class="card-description">Discriminatory test cases found in local search.</div>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px;">
            <div class="chart-container">
                <div class="chart-title">QID Distribution Histogram</div>
                <div id="qid-histogram" style="height: 320px;"></div>
            </div>
            <div class="chart-container">
                <div class="chart-title">Disparate Impact Gauge</div>
                <div id="disparate-impact-gauge" style="height: 320px;"></div>
            </div>
        </div>
        <div class="chart-container">
            <div class="chart-title">QID Values by Instance (Scatter Plot)</div>
            <div id="qid-chart" style="height: 350px;"></div>
        </div>
    </div>`;
}

function buildLayerSection(
    layerAnalysis: { biased_layer: { sensitivity: number; layer_name: string; neuron_count: number }; all_layers: unknown[] } | null,
    neuronAnalysis: { neuron_idx: number; impact_score: number }[] | null,
): string {
    if (!layerAnalysis) {
        return '';
    }
    return `
    <div class="section">
        <div class="section-header">
            <div class="section-icon" style="background: rgba(156, 39, 176, 0.2); color: var(--accent-purple);">
                ${ICONS.layers}
            </div>
            <h2 class="section-title">Causal Debugging: Layer Analysis</h2>
        </div>
        <div class="cards-grid">
            <div class="card">
                <div class="card-header">
                    <span class="card-label">Most Biased Layer</span>
                    <span class="card-badge ${layerAnalysis.biased_layer.sensitivity > 0.5 ? 'badge-warning' : 'badge-info'}">
                        ${layerAnalysis.biased_layer.sensitivity > 0.5 ? 'High Sensitivity' : 'Moderate'}
                    </span>
                </div>
                <div class="card-value">${layerAnalysis.biased_layer.layer_name}</div>
                <div class="card-description">
                    Contains ${layerAnalysis.biased_layer.neuron_count} neurons with sensitivity score of ${layerAnalysis.biased_layer.sensitivity.toFixed(4)}
                </div>
            </div>
        </div>
        <div class="chart-container">
            <div class="chart-title">Layer-Wise Bias Sensitivity</div>
            <div id="layer-chart" style="height: 300px;"></div>
        </div>
    </div>

    ${neuronAnalysis ? `
    <div class="section">
        <div class="section-header">
            <div class="section-icon" style="background: rgba(244, 67, 54, 0.2); color: var(--accent-red);">
                ${ICONS.zap}
            </div>
            <h2 class="section-title">Neuron-Level Localization</h2>
        </div>
        <div class="cards-grid">
            <div class="card" style="grid-column: span 2;">
                <div class="card-header"><span class="card-label">Top Biased Neurons</span></div>
                <div class="neuron-list">
                    ${neuronAnalysis
                        .map(
                            (n, idx) => `
                        <div class="neuron-item">
                            <div class="neuron-info">
                                <div class="neuron-rank">${idx + 1}</div>
                                <span class="neuron-name">Neuron ${n.neuron_idx}</span>
                            </div>
                            <span class="neuron-score">Impact: ${n.impact_score.toFixed(4)}</span>
                        </div>`,
                        )
                        .join('')}
                </div>
            </div>
        </div>
        <div class="chart-container">
            <div class="chart-title">Neuron Impact Scores</div>
            <div id="neuron-chart" style="height: 300px;"></div>
        </div>
    </div>` : ''}`;
}

function buildShapSection(explanationsData: { shap?: Record<string, unknown>; lime?: Record<string, unknown> } | null): string {
    if (!explanationsData?.shap) {
        return '';
    }
    return `
    <div class="section">
        <div class="section-header">
            <div class="section-icon" style="background: rgba(255, 152, 0, 0.2); color: var(--accent-orange);">
                ${ICONS.trendingUp}
            </div>
            <h2 class="section-title">SHAP Feature Importance</h2>
        </div>
        <div class="interpretation-box" style="margin-bottom: 16px;">
            <div class="interpretation-title">${ICONS.info} What are SHAP Values?</div>
            <div class="interpretation-content">
                <strong>SHAP (SHapley Additive exPlanations)</strong> uses game theory to compute the contribution of each feature to the model's prediction.
                Values are computed in <strong>log-odds space</strong> for more meaningful feature attributions.
                Higher bars indicate features with more influence. Features related to protected attributes may indicate bias pathways.
                In the beeswarm plot, each dot represents one instance. Color indicates the feature value (blue = low, red = high).
                Dots to the right push the prediction toward favorable, dots to the left push toward unfavorable.
            </div>
        </div>
        <div class="chart-container">
            <div class="chart-title">Global Feature Importance (Mean |SHAP Value|)</div>
            <div id="shap-global-chart" style="height: 400px;"></div>
        </div>
        <div class="chart-container" style="margin-top: 16px;">
            <div class="chart-title">SHAP Summary (Beeswarm Plot)</div>
            <div id="shap-beeswarm-chart" style="height: 400px;"></div>
        </div>
    </div>`;
}

function buildLimeSection(explanationsData: { shap?: Record<string, unknown>; lime?: Record<string, unknown> } | null): string {
    if (!explanationsData?.lime) {
        return '';
    }
    return `
    <div class="section">
        <div class="section-header">
            <div class="section-icon" style="background: rgba(79, 195, 247, 0.2); color: var(--accent-blue);">
                ${ICONS.scatter}
            </div>
            <h2 class="section-title">LIME Local Explanations</h2>
        </div>
        <div class="interpretation-box" style="margin-bottom: 16px;">
            <div class="interpretation-title">${ICONS.info} What is LIME?</div>
            <div class="interpretation-content">
                <strong>LIME (Local Interpretable Model-agnostic Explanations)</strong> explains individual predictions by learning a simple local model.
                It shows which features push the prediction toward or away from each class. Protected attributes appearing prominently suggest discrimination.
            </div>
        </div>
        <div class="chart-container">
            <div class="chart-title">Aggregated Feature Importance (LIME)</div>
            <div id="lime-global-chart" style="height: 400px;"></div>
        </div>
    </div>`;
}

function buildInterpretationSection(): string {
    return `
    <div class="section">
        <div class="section-header">
            <div class="section-icon" style="background: rgba(79, 195, 247, 0.2); color: var(--accent-blue);">
                ${ICONS.lightbulb}
            </div>
            <h2 class="section-title">Understanding the Results</h2>
        </div>
        <div class="interpretation-box">
            <div class="interpretation-title">${ICONS.info} What is QID (Quantitative Individual Discrimination)?</div>
            <div class="interpretation-content">
                QID measures how many <strong>bits of protected information</strong> (e.g., gender, race, age) the model uses to make its predictions.
                A value of <strong>0 bits</strong> means the model is perfectly fair and doesn't use any protected attributes.
                <strong>Higher values indicate more bias</strong> - the model is learning to discriminate based on protected characteristics.
            </div>
        </div>
        <div class="interpretation-box" style="margin-top: 16px;">
            <div class="interpretation-title">${ICONS.info} The 80% Rule (Four-Fifths Rule)</div>
            <div class="interpretation-content">
                The <strong>disparate impact ratio</strong> should be >= <strong>0.8 (80%)</strong> to comply with legal standards in employment and lending.
                This means the selection rate for a protected group should be at least 80% of the rate for the most favored group.
                <strong>Values below 0.8 may indicate legally actionable discrimination</strong> and could expose your organization to regulatory scrutiny.
            </div>
        </div>
    </div>`;
}
