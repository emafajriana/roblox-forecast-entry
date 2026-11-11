import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

interface Game {
  rank: number;
  universe_id: number;
}

interface Read {
  at: string;
  up_and_coming?: Game[];
  details?: Record<string, { created?: string | null } | null>;
}

interface Task {
  places: number;
  output: string;
  window: {
    from: string;
    until: string;
  };
}

interface Observation {
  universeId: number;
  rank: number;
  weight: number;
}

interface Probability {
  universe_id: number;
  place: number;
  p: number;
}

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const HALF_LIFE_HOURS = 21;
const MAX_CANDIDATE_RANK = 24;
const RECENCY_BOOST = 1.5;
const KERNEL_SIGMA = 1.9;
const TARGET_PLACE_MASS = 0.84;
const LAST_SEEN_GRACE_HOURS = 12;
const LAST_SEEN_DECAY = 0.6;
const ELIGIBILITY_DAYS = 28;
const MAX_ROWS_PER_PLACE = 100;
const PROBABILITY_DECIMALS = 6;

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as Task;
const history = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((line) => line.trim().length > 0)
  .map((line) => JSON.parse(line) as Read);

if (history.length === 0) {
  throw new Error('history.ndjson contains no reads');
}

const readTimes = history.map((read) => Date.parse(read.at));
const endTime = readTimes[readTimes.length - 1]!;
const windowStart = Date.parse(task.window.from);
const windowEnd = Date.parse(task.window.until);
const windowHours = (windowEnd - windowStart) / (60 * 60 * 1000);

if (![endTime, windowStart, windowEnd].every(Number.isFinite) || windowHours <= 0) {
  throw new Error('History or task window contains an invalid timestamp');
}

const observations: Observation[] = [];
const lastSeen = new Map<number, number>();
const latestCreated = new Map<number, { readTime: number; createdTime: number }>();

for (let readIndex = 0; readIndex < history.length; readIndex++) {
  const read = history[readIndex]!;
  const readTime = readTimes[readIndex]!;
  const ageHours = (endTime - readTime) / (60 * 60 * 1000);
  const baseWeight = 0.5 ** (ageHours / HALF_LIFE_HOURS);
  const weight = baseWeight * (readIndex === history.length - 1 ? RECENCY_BOOST : 1);

  for (const game of read.up_and_coming ?? []) {
    lastSeen.set(game.universe_id, Math.max(lastSeen.get(game.universe_id) ?? -Infinity, readTime));
    if (game.rank <= MAX_CANDIDATE_RANK) {
      observations.push({ universeId: game.universe_id, rank: game.rank, weight });
    }
  }

  for (const [id, details] of Object.entries(read.details ?? {})) {
    if (details?.created == null) continue;
    const createdTime = Date.parse(details.created);
    if (!Number.isFinite(createdTime)) continue;
    const universeId = Number(id);
    const previous = latestCreated.get(universeId);
    if (!previous || readTime >= previous.readTime) {
      latestCreated.set(universeId, { readTime, createdTime });
    }
  }
}

const candidateIds = new Set(observations.map((observation) => observation.universeId));
const rawScores = new Map<number, number[]>();

for (const universeId of candidateIds) {
  rawScores.set(universeId, Array(task.places).fill(0) as number[]);
}

for (const observation of observations) {
  const scores = rawScores.get(observation.universeId)!;
  for (let place = 1; place <= task.places; place++) {
    const distance = observation.rank - place;
    scores[place - 1]! += observation.weight * Math.exp(-(distance * distance) / (2 * KERNEL_SIGMA ** 2));
  }
}

function gameScale(universeId: number): number {
  let scale = 1;
  const lastSeenTime = lastSeen.get(universeId);
  if (lastSeenTime !== undefined) {
    const gapHours = (endTime - lastSeenTime) / (60 * 60 * 1000);
    if (gapHours > LAST_SEEN_GRACE_HOURS) {
      scale *= LAST_SEEN_DECAY ** (gapHours / 24);
    }
  }

  const created = latestCreated.get(universeId);
  if (created) {
    const eligibilityEnd = created.createdTime + ELIGIBILITY_DAYS * 24 * 60 * 60 * 1000;
    if (eligibilityEnd <= windowStart) return 0;
    if (eligibilityEnd < windowEnd) {
      scale *= Math.max(0, Math.min(1, (eligibilityEnd - windowStart) / (windowHours * 60 * 60 * 1000)));
    }
  }
  return scale;
}

const scales = new Map<number, number>();
for (const universeId of candidateIds) scales.set(universeId, gameScale(universeId));

const probabilities: Probability[] = [];
const probabilityMass = new Map<number, number>();
const weightedPlaceTotal = new Map<number, number>();

for (let place = 1; place <= task.places; place++) {
  const placeScores = [...rawScores.entries()]
    .map(([universeId, scores]) => ({
      universeId,
      score: scores[place - 1]! * scales.get(universeId)!,
    }))
    .filter((entry) => entry.score > 0)
    .sort((a, b) => b.score - a.score || a.universeId - b.universeId)
    .slice(0, MAX_ROWS_PER_PLACE);
  const totalScore = placeScores.reduce((sum, entry) => sum + entry.score, 0);
  if (totalScore === 0) continue;

  const placeProbabilities = placeScores
    .map(({ universeId, score }) => ({
      universe_id: universeId,
      place,
      p: Number(((score / totalScore) * TARGET_PLACE_MASS).toFixed(PROBABILITY_DECIMALS)),
    }))
    .filter((row) => row.p > 0);
  const roundedTotal = placeProbabilities.reduce((sum, row) => sum + row.p, 0);
  const correction = roundedTotal > 1 ? 1 / roundedTotal : 1;

  for (const row of placeProbabilities) {
    row.p *= correction;
    probabilities.push(row);
    probabilityMass.set(row.universe_id, (probabilityMass.get(row.universe_id) ?? 0) + row.p);
    weightedPlaceTotal.set(row.universe_id, (weightedPlaceTotal.get(row.universe_id) ?? 0) + row.p * place);
  }
}

const top10 = [...probabilityMass.keys()]
  .sort((a, b) => {
    const expectedPlaceA = weightedPlaceTotal.get(a)! / probabilityMass.get(a)!;
    const expectedPlaceB = weightedPlaceTotal.get(b)! / probabilityMass.get(b)!;
    return expectedPlaceA - expectedPlaceB
      || probabilityMass.get(b)! - probabilityMass.get(a)!
      || a - b;
  })
  .slice(0, 10);

if (top10.length !== 10) {
  throw new Error(`Expected at least 10 forecast candidates, found ${top10.length}`);
}

mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top10 })}\n`);
