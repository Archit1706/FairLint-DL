import { getStyles } from './styles';
import { ICONS } from './icons';
import { calculateFairnessScore, getFairnessStatus } from './scoring';
import { getChartsScript } from './charts';
import {
    AnalysisResults,
    TrainingData,
    QidMetrics,
    SearchResults,
    LayerAnalysis,
    NeuronAnalysis,
    ShapData,
    LimeData,
    ActivationsData,
    AnalysisMetadata,
} from './types';

export function getWebviewHtml(results: AnalysisResults): string {
    const qidMetrics = results.analysis.qid_metrics;
    const searchResults = results.search.search_results;
    const layerAnalysis = (results.debug?.layer_analysis as LayerAnalysis) || null;
    const neuronAnalysis = (results.debug?.neuron_analysis as NeuronAnalysis[]) || null;
    const activationsData = (results.activations as ActivationsData) || null;
    const explanationsData = results.explanations || null;
    const metadata = results.metadata;
    const training = results.training;

    const fairnessScore = calculateFairnessScore(qidMetrics);
    const fairnessStatus = getFairnessStatus(fairnessScore);

    const headScript = `
    <script>
        var _vscodeApi = (function() { try { return acquireVsCodeApi(); } catch(e) { return null; } })();
        function downloadJson() {
            if (!_vscodeApi) return;
            try { _vscodeApi.postMessage({ command: 'saveJson' }); } catch (e) { console.error('Download JSON error:', e); }
        }
    </script>`;

    const chartsScript = getChartsScript(searchResults, qidMetrics, layerAnalysis, neuronAnalysis, activationsData, explanationsData);

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
    ${buildPipelineOverview(training, metadata, qidMetrics, searchResults)}
    ${buildTrainingSection(training)}
    ${buildActivationsSection(activationsData)}
    ${buildQidSection(qidMetrics, metadata)}
    ${buildSearchSection(searchResults, qidMetrics)}
    ${buildLayerSection(layerAnalysis, neuronAnalysis, training)}
    ${buildShapSection(explanationsData?.shap as ShapData | null, training)}
    ${buildLimeSection(explanationsData?.lime as LimeData | null, training)}
    ${chartsScript}
</body>
</html>`;
}

function buildHeader(
    metadata: AnalysisMetadata | undefined,
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

function buildPipelineOverview(
    training: TrainingData,
    metadata: AnalysisMetadata | undefined,
    qidMetrics: QidMetrics,
    searchResults: SearchResults,
): string {
    const ds = training.dataset_info;
    const hist = training.training_history;
    const timings = metadata?.stepTimings || {};

    // Class distribution text
    let classDistText = '';
    if (ds?.class_distribution) {
        const entries = Object.entries(ds.class_distribution);
        const total = entries.reduce((sum, [, count]) => sum + count, 0);
        classDistText = entries
            .map(([cls, count]) => `Class ${cls}: ${count} (${((count / total) * 100).toFixed(1)}%)`)
            .join(', ');
    }

    // Protected attributes info
    const protectedAttrs = metadata?.protectedFeatures || training.protected_features;
    const protectedAttrInfo = ds?.protected_attr_info || {};

    return `
    <div class="section">
        <div class="section-header">
            <div class="section-icon" style="background: rgba(79, 195, 247, 0.2); color: var(--accent-blue);">
                ${ICONS.info}
            </div>
            <h2 class="section-title">Analysis Pipeline Overview</h2>
        </div>

        <div class="interpretation-box" style="margin-bottom: 16px;">
            <div class="interpretation-title">${ICONS.file} Dataset Summary</div>
            <div class="interpretation-content">
                Analyzed <strong>${metadata?.file || 'dataset'}</strong> with
                <strong>${ds?.num_total || 'N/A'} total instances</strong> and
                <strong>${ds?.num_features || 'N/A'} features</strong>.
                The target variable is <strong>"${metadata?.labelColumn || 'N/A'}"</strong>.
                ${classDistText ? `<br>Class distribution: <strong>${classDistText}</strong>.` : ''}
                ${ds?.num_total ? `<br>Data split: ${ds.num_train} training, ${ds.num_val} validation, ${ds.num_test} test instances.` : ''}
            </div>
        </div>

        <div class="interpretation-box" style="margin-bottom: 16px;">
            <div class="interpretation-title">${ICONS.layers} Protected Attributes</div>
            <div class="interpretation-content">
                ${protectedAttrs.length} protected attribute${protectedAttrs.length !== 1 ? 's' : ''} selected for fairness analysis:
                <strong>${protectedAttrs.join(', ')}</strong>.
                ${protectedAttrs.map((attr) => {
                    const info = protectedAttrInfo[attr];
                    return `<br>&bull; <strong>${attr}</strong> (feature index ${info?.index ?? 'N/A'}):
                        The analysis tests whether changing this attribute's value (e.g., from one group to another)
                        causes different model predictions, indicating potential discrimination.`;
                }).join('')}
                <br><br>During QID analysis, the model's behavior is compared when protected attribute values are set to
                <strong>0.0</strong> (normalized baseline group) vs <strong>1.0</strong> (normalized comparison group).
                The difference in predictions reveals how much the model relies on these attributes.
            </div>
        </div>

        <div class="interpretation-box" style="margin-bottom: 16px;">
            <div class="interpretation-title">${ICONS.barChart} What Happened During Analysis</div>
            <div class="interpretation-content">
                The 6-step analysis pipeline was executed as follows:
                <br><br>
                <strong>Step 1 - Model Training</strong> (${timings.training || '?'}s):
                A deep neural network with architecture [${(metadata?.hiddenLayers || training.hidden_layers || []).join(' &rarr; ')} &rarr; 2]
                was trained for <strong>${hist?.epochs_trained || metadata?.epochs || '?'} epochs</strong>.
                Final accuracy: <strong>${training.accuracy.toFixed(1)}%</strong>
                (train loss: ${hist?.final_train_loss?.toFixed(4) || 'N/A'}, val loss: ${hist?.final_val_loss?.toFixed(4) || 'N/A'}).
                <br><br>
                <strong>Step 2 - Internal Space Visualization</strong> (${timings.activations || '?'}s):
                Extracted activations from each hidden layer and projected them to 2D using PCA for
                <strong>${metadata?.maxSamples || 'N/A'}</strong> test instances.
                <br><br>
                <strong>Step 3 - QID Fairness Analysis</strong> (${timings.qidAnalysis || '?'}s):
                Computed Quantitative Individual Discrimination (QID) for <strong>${qidMetrics.num_analyzed || metadata?.maxSamples || 'N/A'}</strong> instances
                by measuring how many bits of protected information the model uses in its decisions.
                <br><br>
                <strong>Step 4 - Discriminatory Instance Search</strong> (${timings.search || '?'}s):
                Ran ${metadata?.globalIterations || 'N/A'} global iterations + ${metadata?.localNeighbors || 'N/A'} local neighbors
                to generate <strong>${searchResults.num_found} discriminatory test cases</strong> with best QID = ${searchResults.best_qid.toFixed(4)} bits.
                <br><br>
                <strong>Step 5 - Causal Debugging</strong> (${timings.debug || '?'}s):
                Localized bias to specific layers and neurons using causal intervention analysis.
                <br><br>
                <strong>Step 6 - LIME &amp; SHAP Explanations</strong> (${timings.explain || '?'}s):
                Computed feature importance for <strong>${metadata?.numExplainInstances || 10} instances</strong>
                using SHAP (Shapley values in log-odds space) and LIME (local linear approximation).
            </div>
        </div>
    </div>`;
}

function buildTrainingSection(training: TrainingData): string {
    const hist = training.training_history;
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
            ${hist ? `
            <div class="card">
                <div class="card-header">
                    <span class="card-label">Training Summary</span>
                </div>
                <div class="card-value">${hist.epochs_trained}<span class="card-unit">epochs</span></div>
                <div class="card-description">
                    Train acc: ${hist.final_train_acc.toFixed(1)}% | Val acc: ${hist.final_val_acc.toFixed(1)}%<br>
                    Train loss: ${hist.final_train_loss.toFixed(4)} | Val loss: ${hist.final_val_loss.toFixed(4)}
                </div>
            </div>` : ''}
        </div>
    </div>`;
}

function buildActivationsSection(activationsData: ActivationsData | null): string {
    if (!activationsData) {
        return '';
    }
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
                Each chart shows the <strong>${activationsData.method.toUpperCase()}</strong> 2D projection of layer activations for <strong>${activationsData.num_samples}</strong> test instances.
                Points are colored by <strong>prediction label</strong>. Clusters that separate by protected attribute indicate the model is learning to encode protected information at that layer.
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 16px;">
            ${activationsData.layers
                .map(
                    (layer, idx) => `
                <div class="chart-container">
                    <div class="chart-title">${layer.layer_name} Activations (${activationsData.method.toUpperCase()})</div>
                    <div id="activation-layer-${idx}" style="height: 350px;"></div>
                </div>`,
                )
                .join('')}
        </div>
    </div>`;
}

function buildQidSection(qidMetrics: QidMetrics, metadata: AnalysisMetadata | undefined): string {
    // Dynamic interpretation based on actual values
    const meanQid = qidMetrics.mean_qid;
    const pctDisc = qidMetrics.pct_discriminatory;
    const di = qidMetrics.mean_disparate_impact;
    const numAnalyzed = qidMetrics.num_analyzed || metadata?.maxSamples || 0;

    let qidInterpretation = '';
    if (meanQid < 0.01) {
        qidInterpretation = `The model shows <strong>very low discrimination</strong> with a mean QID of only ${meanQid.toFixed(4)} bits. This means the model uses almost no protected information when making predictions. Only ${pctDisc.toFixed(1)}% of the ${numAnalyzed} analyzed instances showed any discriminatory behavior.`;
    } else if (meanQid < 0.1) {
        qidInterpretation = `The model shows <strong>mild discrimination</strong> with a mean QID of ${meanQid.toFixed(4)} bits. While not zero, this level suggests the model makes <strong>slight use</strong> of protected attributes. ${pctDisc.toFixed(1)}% of instances (${qidMetrics.num_discriminatory} out of ${numAnalyzed}) showed discriminatory behavior above the 0.1-bit threshold.`;
    } else if (meanQid < 0.5) {
        qidInterpretation = `The model shows <strong>moderate discrimination</strong> with a mean QID of ${meanQid.toFixed(4)} bits. This indicates the model is <strong>meaningfully using protected attributes</strong> in its decision-making. ${pctDisc.toFixed(1)}% of instances exhibited discriminatory behavior, which warrants further investigation and possible mitigation.`;
    } else {
        qidInterpretation = `The model shows <strong>significant discrimination</strong> with a mean QID of ${meanQid.toFixed(4)} bits. This is a <strong>serious concern</strong> - the model is heavily relying on protected attributes. ${pctDisc.toFixed(1)}% of instances are discriminatory. The worst case reaches ${qidMetrics.max_qid.toFixed(4)} bits. <strong>Retraining with fairness constraints is strongly recommended.</strong>`;
    }

    let diInterpretation = '';
    if (di >= 0.8) {
        diInterpretation = `The disparate impact ratio of <strong>${di.toFixed(3)}</strong> <strong>passes</strong> the four-fifths (80%) rule, indicating that protected groups receive favorable outcomes at rates comparable to the majority group.`;
    } else if (di >= 0.6) {
        diInterpretation = `The disparate impact ratio of <strong>${di.toFixed(3)}</strong> <strong>falls below</strong> the 80% threshold. Protected groups are receiving favorable outcomes at only <strong>${(di * 100).toFixed(1)}%</strong> of the rate of the majority group. This may constitute a legal violation in employment or lending contexts.`;
    } else {
        diInterpretation = `The disparate impact ratio of <strong>${di.toFixed(3)}</strong> indicates <strong>severe disparate impact</strong>. Protected groups receive favorable outcomes at less than ${(di * 100).toFixed(1)}% of the majority group's rate. This is a <strong>critical fairness violation</strong> that likely requires model revision.`;
    }

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
                    <span class="card-badge ${meanQid > 1.0 ? 'badge-warning' : 'badge-success'}">${meanQid > 1.0 ? 'High' : 'Low'}</span>
                </div>
                <div class="card-value">${meanQid.toFixed(4)}<span class="card-unit">bits</span></div>
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
                    <span class="card-badge ${pctDisc > 10 ? 'badge-warning' : 'badge-success'}">${pctDisc > 10 ? 'Concerning' : 'Acceptable'}</span>
                </div>
                <div class="card-value">${qidMetrics.num_discriminatory}<span class="card-unit">(${pctDisc.toFixed(1)}%)</span></div>
                <div class="card-description">Instances with QID > 0.1 bits showing potential bias.</div>
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${Math.min(pctDisc, 100)}%; background: ${pctDisc > 10 ? 'var(--accent-orange)' : 'var(--accent-green)'};"></div>
                    </div>
                    <div class="progress-labels"><span>0%</span><span>10% threshold</span><span>100%</span></div>
                </div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-label">Disparate Impact Ratio</span>
                    <span class="card-badge ${di < 0.8 ? 'badge-danger' : 'badge-success'}">${di < 0.8 ? 'Violation' : 'Compliant'}</span>
                </div>
                <div class="card-value">${di.toFixed(3)}</div>
                <div class="card-description">Ratio should be >= 0.8 for legal compliance.</div>
                <div class="compliance-box ${di >= 0.8 ? 'compliance-pass' : 'compliance-fail'}">
                    <div class="compliance-header">
                        ${di >= 0.8 ? '&#10003; Passes 80% Rule' : '&#10007; Violates 80% Rule'}
                    </div>
                    <div class="compliance-text">
                        ${di >= 0.8
                            ? 'Model meets legal fairness thresholds for hiring/lending decisions.'
                            : 'Model may exhibit legally actionable discrimination. Consider retraining with fairness constraints.'}
                    </div>
                </div>
            </div>
        </div>
        <div class="interpretation-box" style="margin-top: 16px;">
            <div class="interpretation-title">${ICONS.lightbulb} Interpretation for This Dataset</div>
            <div class="interpretation-content">
                ${qidInterpretation}
                <br><br>${diInterpretation}
            </div>
        </div>
    </div>`;
}

function buildSearchSection(searchResults: SearchResults, _qidMetrics: QidMetrics): string {
    const bestQid = searchResults.best_qid;
    const numFound = searchResults.num_found;

    let searchInterpretation = '';
    if (bestQid < 0.01) {
        searchInterpretation = `The search algorithm was <strong>unable to find strongly discriminatory instances</strong> (best QID: ${bestQid.toFixed(4)} bits). This is a positive sign suggesting the model has limited exploitable bias pathways. ${numFound} test cases were generated but all show minimal discrimination.`;
    } else if (bestQid < 0.1) {
        searchInterpretation = `The search found <strong>mild discriminatory behavior</strong> with the worst case at ${bestQid.toFixed(4)} bits. ${numFound} discriminatory instances were generated through local perturbation. While the discrimination level is relatively low, these instances reveal the specific input regions where the model is most vulnerable to bias.`;
    } else {
        searchInterpretation = `The search discovered <strong>notable discriminatory instances</strong> with the worst case reaching ${bestQid.toFixed(4)} bits. ${numFound} test cases demonstrate that specific input combinations cause the model to behave differently based on protected attributes. The histogram below shows how discrimination varies across these instances.`;
    }

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
                <div class="card-value">${bestQid.toFixed(4)}<span class="card-unit">bits</span></div>
                <div class="card-description">Maximum discrimination discovered via gradient search.</div>
            </div>
            <div class="card">
                <div class="card-header"><span class="card-label">Instances Generated</span></div>
                <div class="card-value">${numFound}</div>
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
        <div class="interpretation-box" style="margin-top: 16px;">
            <div class="interpretation-title">${ICONS.lightbulb} Search Results Interpretation</div>
            <div class="interpretation-content">${searchInterpretation}</div>
        </div>
    </div>`;
}

function buildLayerSection(
    layerAnalysis: LayerAnalysis | null,
    neuronAnalysis: NeuronAnalysis[] | null,
    _training: TrainingData,
): string {
    if (!layerAnalysis) {
        return '';
    }

    const biased = layerAnalysis.biased_layer;
    const allLayers = layerAnalysis.all_layers || [];
    const totalLayers = allLayers.length;

    // Dynamic interpretation
    let layerInterpretation = `The causal analysis identified <strong>${biased.layer_name}</strong> as the most biased layer
        with a sensitivity score of <strong>${biased.sensitivity.toFixed(4)}</strong>.
        This layer contains <strong>${biased.neuron_count} neurons</strong> and is `;

    if (biased.layer_idx === 0) {
        layerInterpretation += `the <strong>first hidden layer</strong>, suggesting bias enters the model early and propagates through subsequent layers. Early-layer bias typically means the model captures demographic patterns directly from input features.`;
    } else if (biased.layer_idx === totalLayers - 1) {
        layerInterpretation += `the <strong>last layer before output</strong>, suggesting bias accumulates through the network and manifests most strongly at the decision boundary. This pattern often occurs when multiple features interact to create discriminatory behavior.`;
    } else {
        layerInterpretation += `in the <strong>middle of the network</strong> (layer ${biased.layer_idx + 1} of ${totalLayers}). Middle-layer bias suggests the model forms intermediate representations that encode protected information.`;
    }

    let neuronInterpretation = '';
    if (neuronAnalysis && neuronAnalysis.length > 0) {
        const topNeuron = neuronAnalysis[0];
        const topImpact = topNeuron.impact_score;
        neuronInterpretation = `
            <br><br>At the neuron level, <strong>Neuron ${topNeuron.neuron_idx}</strong> has the highest bias impact score of <strong>${topImpact.toFixed(4)}</strong>.
            ${neuronAnalysis.length > 1 ? `The top ${Math.min(neuronAnalysis.length, 5)} biased neurons account for the majority of the layer's discriminatory behavior.` : ''}
            These specific neurons could be targeted for debiasing techniques such as neuron pruning or fine-tuning.`;
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
                    <span class="card-badge ${biased.sensitivity > 0.5 ? 'badge-warning' : 'badge-info'}">
                        ${biased.sensitivity > 0.5 ? 'High Sensitivity' : 'Moderate'}
                    </span>
                </div>
                <div class="card-value">${biased.layer_name}</div>
                <div class="card-description">
                    Contains ${biased.neuron_count} neurons with sensitivity score of ${biased.sensitivity.toFixed(4)}
                </div>
            </div>
        </div>
        <div class="chart-container">
            <div class="chart-title">Layer-Wise Bias Sensitivity</div>
            <div id="layer-chart" style="height: 300px;"></div>
        </div>
        <div class="interpretation-box" style="margin-top: 16px;">
            <div class="interpretation-title">${ICONS.lightbulb} Layer Analysis Interpretation</div>
            <div class="interpretation-content">${layerInterpretation}${neuronInterpretation}</div>
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

function buildShapSection(shapData: ShapData | null, training: TrainingData): string {
    if (!shapData) {
        return '';
    }

    // Dynamic interpretation based on SHAP values
    const importance = shapData.global_importance;
    const names = shapData.feature_names;
    const protectedAttrs = training.protected_features;

    // Sort features by importance
    const sorted = importance
        .map((val: number, idx: number) => ({ name: names[idx], importance: val, idx }))
        .sort((a: { importance: number }, b: { importance: number }) => b.importance - a.importance);

    const topFeature = sorted[0];
    const topProtected = sorted.filter((f: { name: string }) => protectedAttrs.includes(f.name));

    let shapInterpretation = `Across <strong>${shapData.num_explained} explained instances</strong>, the most influential feature is <strong>${topFeature.name}</strong> with a mean |SHAP value| of <strong>${topFeature.importance.toFixed(4)}</strong> in log-odds space.`;

    if (topProtected.length > 0) {
        const topP = topProtected[0];
        const rank = sorted.findIndex((f: { name: string }) => f.name === topP.name) + 1;
        if (rank <= 3) {
            shapInterpretation += ` <br><br><strong style="color: var(--accent-red);">Fairness concern:</strong> The protected attribute <strong>"${topP.name}"</strong> ranks <strong>#${rank}</strong> in feature importance (SHAP value: ${topP.importance.toFixed(4)}). This means the model is <strong>significantly using this protected attribute</strong> to make predictions, which is a direct indicator of potential discrimination.`;
        } else {
            shapInterpretation += ` <br><br>The protected attribute <strong>"${topP.name}"</strong> ranks #${rank} out of ${names.length} features (SHAP value: ${topP.importance.toFixed(4)}). Its relatively low ranking suggests the model <strong>does not heavily rely</strong> on this protected attribute for predictions.`;
        }
    }

    shapInterpretation += `<br><br>The top 3 most influential features are: <strong>${sorted.slice(0, 3).map((f: { name: string; importance: number }) => `${f.name} (${f.importance.toFixed(4)})`).join('</strong>, <strong>')}</strong>.`;

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
                Higher bars indicate features with more influence. In the beeswarm plot, each dot represents one instance.
                Color indicates the feature value (blue = low, red = high).
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
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px;">
            <div class="chart-container">
                <div class="chart-title">SHAP Scatter: ${topFeature.name} (Top Feature)</div>
                <div id="shap-scatter-chart" style="height: 350px;"></div>
            </div>
            <div class="chart-container">
                <div class="chart-title">SHAP Heatmap (Instances vs Features)</div>
                <div id="shap-heatmap-chart" style="height: 350px;"></div>
            </div>
        </div>
        <div class="interpretation-box" style="margin-top: 16px;">
            <div class="interpretation-title">${ICONS.lightbulb} SHAP Interpretation for This Dataset</div>
            <div class="interpretation-content">${shapInterpretation}</div>
        </div>
    </div>`;
}

function buildLimeSection(limeData: LimeData | null, training: TrainingData): string {
    if (!limeData) {
        return '';
    }

    // Dynamic interpretation
    const importance = limeData.aggregated_importance;
    const names = limeData.feature_names;
    const protectedAttrs = training.protected_features;

    const sorted = importance
        .map((val: number, idx: number) => ({ name: names[idx], importance: val, idx }))
        .sort((a: { importance: number }, b: { importance: number }) => b.importance - a.importance);

    const topFeature = sorted[0];
    const topProtected = sorted.filter((f: { name: string }) => protectedAttrs.includes(f.name));

    let limeInterpretation = `LIME analyzed <strong>${limeData.num_explained} instances</strong> by perturbing inputs locally and fitting linear models. The most influential feature is <strong>${topFeature.name}</strong> with an aggregated importance of <strong>${topFeature.importance.toFixed(4)}</strong>.`;

    if (topProtected.length > 0) {
        const topP = topProtected[0];
        const rank = sorted.findIndex((f: { name: string }) => f.name === topP.name) + 1;
        limeInterpretation += ` The protected attribute <strong>"${topP.name}"</strong> ranks #${rank} in LIME importance.`;

        if (rank <= 3) {
            limeInterpretation += ` This <strong>corroborates the SHAP findings</strong> - the model locally relies on protected information for individual predictions.`;
        }
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
        <div class="interpretation-box" style="margin-top: 16px;">
            <div class="interpretation-title">${ICONS.lightbulb} LIME Interpretation for This Dataset</div>
            <div class="interpretation-content">${limeInterpretation}</div>
        </div>
    </div>`;
}
