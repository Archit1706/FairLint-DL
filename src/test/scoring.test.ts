import { test } from 'node:test';
import * as assert from 'node:assert';
import { calculateFairnessScore, getFairnessStatus } from '../webview/scoring';

test('perfect fairness scores 100', () => {
    const score = calculateFairnessScore({
        mean_qid: 0,
        max_qid: 0,
        pct_discriminatory: 0,
        mean_disparate_impact: 1.0,
    });
    assert.strictEqual(score, 100);
});

test('severe discrimination scores low and is bounded to [0,100]', () => {
    const score = calculateFairnessScore({
        mean_qid: 1.0,
        max_qid: 1.0,
        pct_discriminatory: 100,
        mean_disparate_impact: 0.3,
    });
    assert.ok(score >= 0 && score < 40, `expected low score, got ${score}`);
});

test('lower disparate impact never increases the score', () => {
    const base = { mean_qid: 0.5, max_qid: 0.8, pct_discriminatory: 50 };
    const better = calculateFairnessScore({ ...base, mean_disparate_impact: 0.9 });
    const worse = calculateFairnessScore({ ...base, mean_disparate_impact: 0.4 });
    assert.ok(worse <= better, `worse DI (${worse}) should not exceed better DI (${better})`);
});

test('group fairness penalty lowers the score', () => {
    const qid = { mean_qid: 0.3, max_qid: 0.5, pct_discriminatory: 20, mean_disparate_impact: 0.9 };
    const withoutGroups = calculateFairnessScore(qid);
    const withGroups = calculateFairnessScore(qid, [
        {
            demographic_parity: { difference: 0.4 },
            equalized_odds: { max_difference: 0.4 },
        } as never,
    ]);
    assert.ok(withGroups < withoutGroups);
});

test('status thresholds map score to label', () => {
    assert.strictEqual(getFairnessStatus(85).label, 'Good');
    assert.strictEqual(getFairnessStatus(70).label, 'Needs Review');
    assert.strictEqual(getFairnessStatus(40).label, 'Concerning');
});
