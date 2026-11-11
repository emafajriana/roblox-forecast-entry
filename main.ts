import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

interface Game {
  rank: number;
  universe_id: number;
}

interface Read {
  at: string;
  up_and_coming?: Game[];
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
  ageHours: number;
  rank: number;
}

interface Probability {
  universe_id: number;
  place: number;
  p: number;
}

interface Candidate {
  universeId: number;
  x0: number;
  drift: number;
  eligibleUntil: number | null;
}

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const PLACES = 10;
const HOURS = 72;
const SAMPLES = 800;
const PROBABILITY_CUTOFF = 0.0005;
const MAX_ROWS_PER_PLACE = 100;
const ELIGIBILITY_MS = 28 * 24 * 60 * 60 * 1000;
const HOUR_MS = 60 * 60 * 1000;

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
if (![endTime, windowStart, windowEnd].every(Number.isFinite) || windowEnd <= windowStart) {
  throw new Error('History or task window contains an invalid timestamp');
}

const observationsByGame = new Map<number, Observation[]>();
const mostRecent = new Map<number, { rank: number; ageHours: number }>();
const finalReadIds = new Set<number>();
const kernelObservations = new Map<number, Array<{ rank: number; weight: number }>>();
const creationByGame = new Map<number, number>();

const detailsPath = join(INPUT_DIR, 'details.json');
const detailsAvailable = existsSync(detailsPath);
if (detailsAvailable) {
  const details = JSON.parse(readFileSync(detailsPath, 'utf8')) as Record<string, { created?: string | null } | null>;
  for (const [id, metadata] of Object.entries(details)) {
    if (metadata?.created == null) continue;
    const createdAt = Date.parse(metadata.created);
    if (Number.isFinite(createdAt)) creationByGame.set(Number(id), createdAt);
  }
}

for (let readIndex = 0; readIndex < history.length; readIndex++) {
  const read = history[readIndex]!;
  const readTime = readTimes[readIndex]!;
  const ageHours = (endTime - readTime) / HOUR_MS;
  const isFinalRead = readIndex === history.length - 1;
  const kernelWeight = 0.5 ** (ageHours / 18);

  for (const game of read.up_and_coming ?? []) {
    const id = game.universe_id;
    const observations = observationsByGame.get(id) ?? [];
    observations.push({ ageHours, rank: game.rank });
    observationsByGame.set(id, observations);

    const currentMostRecent = mostRecent.get(id);
    if (!currentMostRecent || ageHours <= currentMostRecent.ageHours) {
      mostRecent.set(id, { rank: game.rank, ageHours });
    }

    if (isFinalRead) finalReadIds.add(id);
    if (game.rank <= 25) {
      const kernel = kernelObservations.get(id) ?? [];
      kernel.push({ rank: game.rank, weight: kernelWeight });
      if (isFinalRead) kernel.push({ rank: game.rank, weight: kernelWeight * 2 });
      kernelObservations.set(id, kernel);
    }
  }
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  values.sort((a, b) => a - b);
  const middle = Math.floor(values.length / 2);
  return values.length % 2 === 0 ? (values[middle - 1]! + values[middle]!) / 2 : values[middle]!;
}

const volatilityMeasures: number[] = [];
for (const observations of observationsByGame.values()) {
  observations.sort((a, b) => b.ageHours - a.ageHours);
  for (let index = 1; index < observations.length; index++) {
    const earlier = observations[index - 1]!;
    const later = observations[index]!;
    const deltaHours = earlier.ageHours - later.ageHours;
    if (deltaHours > 0.5) {
      volatilityMeasures.push(Math.abs(Math.log(earlier.rank) - Math.log(later.rank)) / Math.sqrt(deltaHours));
    }
  }
}
const globalVolatility = median(volatilityMeasures);

const kernelScores = new Map<number, number[]>();
for (const [id, observations] of kernelObservations) {
  const scores = Array(PLACES).fill(0) as number[];
  for (const observation of observations) {
    for (let place = 1; place <= PLACES; place++) {
      const distance = observation.rank - place;
      scores[place - 1]! += observation.weight * Math.exp(-(distance * distance) / (2 * 2.0 ** 2));
    }
  }

  if (detailsAvailable) {
    const createdAt = creationByGame.get(id);
    if (createdAt !== undefined) {
      const eligibleUntil = createdAt + ELIGIBILITY_MS;
      const eligibleHours = Math.max(0, Math.min(HOURS, (eligibleUntil - windowStart) / HOUR_MS));
      const eligibilityScale = eligibleHours / HOURS;
      for (let place = 0; place < PLACES; place++) scores[place]! *= eligibilityScale;
    }
  }

  const last = mostRecent.get(id)!;
  const lastSeenScale = 0.3 ** (last.ageHours / 24);
  for (let place = 0; place < PLACES; place++) scores[place]! *= lastSeenScale;
  kernelScores.set(id, scores);
}

const kernelProbabilities = new Map<number, number[]>();
for (let place = 0; place < PLACES; place++) {
  const total = [...kernelScores.values()].reduce((sum, scores) => sum + scores[place]!, 0);
  for (const [id, scores] of kernelScores) {
    const probability = total > 0 ? (scores[place]! / total) * 0.82 : 0;
    const placeProbabilities = kernelProbabilities.get(id) ?? Array(PLACES).fill(0) as number[];
    placeProbabilities[place] = probability;
    kernelProbabilities.set(id, placeProbabilities);
  }
}

function estimateDrift(observations: Observation[]): number {
  const recent = observations.filter((observation) => observation.ageHours <= 120);
  if (recent.length < 3) return 0;

  const points = recent.map((observation) => ({
    time: -observation.ageHours,
    logRank: Math.log(observation.rank),
  }));
  const meanTime = points.reduce((sum, point) => sum + point.time, 0) / points.length;
  const meanLogRank = points.reduce((sum, point) => sum + point.logRank, 0) / points.length;
  let covariance = 0;
  let timeVariance = 0;
  for (const point of points) {
    covariance += (point.time - meanTime) * (point.logRank - meanLogRank);
    timeVariance += (point.time - meanTime) ** 2;
  }
  return timeVariance > 0 ? (covariance / timeVariance) * 0.5 : 0;
}

const candidateIds = new Set<number>(finalReadIds);
for (const [id, observation] of mostRecent) {
  if (observation.ageHours <= 96) candidateIds.add(id);
}

const candidates: Candidate[] = [...observationsByGame.keys()]
  .filter((id) => candidateIds.has(id))
  .map((universeId) => {
    const latest = mostRecent.get(universeId)!;
    const createdAt = detailsAvailable ? creationByGame.get(universeId) : undefined;
    return {
      universeId,
      x0: Math.log(latest.rank) + 0.02 * latest.ageHours,
      drift: estimateDrift(observationsByGame.get(universeId) ?? []),
      eligibleUntil: createdAt === undefined ? null : createdAt + ELIGIBILITY_MS,
    };
  });

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    let value = state += 0x6d2b79f5;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianGenerator(random: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    let first = random();
    while (first === 0) first = random();
    const second = random();
    const magnitude = Math.sqrt(-2 * Math.log(first));
    const angle = 2 * Math.PI * second;
    spare = magnitude * Math.sin(angle);
    return magnitude * Math.cos(angle);
  };
}

const counts = new Map<number, number[]>();
for (const candidate of candidates) counts.set(candidate.universeId, Array(PLACES).fill(0) as number[]);

const normal = gaussianGenerator(mulberry32(4242));
const values: Array<{ candidate: Candidate; value: number }> = [];
for (let sample = 0; sample < SAMPLES; sample++) {
  for (let hour = 1; hour <= HOURS; hour++) {
    values.length = 0;
    const forecastTime = endTime + hour * HOUR_MS;
    const horizon = Math.min(hour, 48);
    const standardDeviation = globalVolatility * 2.0 * Math.sqrt(horizon);

    for (const candidate of candidates) {
      if (candidate.eligibleUntil !== null && forecastTime > candidate.eligibleUntil) continue;
      values.push({
        candidate,
        value: candidate.x0 + candidate.drift * horizon + normal() * standardDeviation,
      });
    }

    values.sort((a, b) => a.value - b.value || a.candidate.universeId - b.candidate.universeId);
    const limit = Math.min(PLACES, values.length);
    for (let place = 0; place < limit; place++) {
      const id = values[place]!.candidate.universeId;
      counts.get(id)![place]!++;
    }
  }
}

const probabilitiesByPlace: Probability[][] = Array.from({ length: PLACES }, () => []);
const allIds = new Set<number>([...kernelScores.keys(), ...counts.keys()]);
for (const id of allIds) {
  const kernel = kernelProbabilities.get(id);
  const monteCarloCounts = counts.get(id);
  for (let place = 0; place < PLACES; place++) {
    const kernelProbability = kernel?.[place] ?? 0;
    const monteCarloProbability = monteCarloCounts
      ? (monteCarloCounts[place]! / (SAMPLES * HOURS)) * 0.84
      : 0;
    const probability = 0.5 * kernelProbability + 0.5 * monteCarloProbability;
    if (probability < PROBABILITY_CUTOFF) continue;
    probabilitiesByPlace[place]!.push({
      universe_id: id,
      place: place + 1,
      p: Number(probability.toFixed(6)),
    });
  }
}

const probabilities: Probability[] = [];
for (const placeRows of probabilitiesByPlace) {
  placeRows.sort((a, b) => b.p - a.p || a.universe_id - b.universe_id);
  placeRows.splice(MAX_ROWS_PER_PLACE);
  let sum = placeRows.reduce((total, row) => total + row.p, 0);
  while (sum > 1 && placeRows.length > 0) {
    const removed = placeRows.pop()!;
    sum -= removed.p;
  }
  probabilities.push(...placeRows);
}

const weightedPlaceSum = new Map<number, number>();
for (const row of probabilities) {
  weightedPlaceSum.set(row.universe_id, (weightedPlaceSum.get(row.universe_id) ?? 0) + row.place * row.p);
}
const top10 = [...weightedPlaceSum.keys()]
  .sort((a, b) => weightedPlaceSum.get(a)! - weightedPlaceSum.get(b)! || a - b)
  .slice(0, PLACES);

if (top10.length !== PLACES) {
  throw new Error(`Expected ${PLACES} forecast candidates with emitted rows, found ${top10.length}`);
}

mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top10 })}\n`);
