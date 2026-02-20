export function calculateFairnessScore(qidMetrics: {
    mean_qid: number;
    max_qid: number;
    pct_discriminatory: number;
    mean_disparate_impact: number;
}): number {
    let score = 100;

    // Penalize for high mean QID (0-2 bits range maps to 0-30 penalty)
    score -= Math.min(qidMetrics.mean_qid * 15, 30);

    // Penalize for discriminatory instances (0-100% maps to 0-30 penalty)
    score -= Math.min(qidMetrics.pct_discriminatory * 0.3, 30);

    // Penalize for low disparate impact (0.8-1.0 is good, below 0.8 is bad)
    if (qidMetrics.mean_disparate_impact < 0.8) {
        score -= (0.8 - qidMetrics.mean_disparate_impact) * 50;
    }

    // Penalize for high max QID
    score -= Math.min(qidMetrics.max_qid * 5, 20);

    return Math.max(0, Math.round(score));
}

export function getFairnessStatus(score: number): { label: string; class: string; color: string } {
    if (score >= 80) {
        return { label: 'Good', class: 'status-good', color: '#4caf50' };
    } else if (score >= 60) {
        return { label: 'Needs Review', class: 'status-warning', color: '#ff9800' };
    } else {
        return { label: 'Concerning', class: 'status-danger', color: '#f44336' };
    }
}
