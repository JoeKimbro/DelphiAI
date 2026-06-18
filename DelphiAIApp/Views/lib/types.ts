// Mirrors DelphiAIApp/Services/event_service.py + fighter_service.py + performance_service.py shapes.

export type Confidence = "HIGH" | "MED" | "LOW" | "TOSS";

export type FightPrediction = {
  fighter1: string;
  fighter2: string;
  f1_prob: number;
  f2_prob: number;
  pick: string;
  pick_pct: number;
  confidence: Confidence;
  prob_source: string | null;
  method: string | null;
  ko_pct: number | null;
  sub_pct: number | null;
  dec_pct: number | null;
  r1_finish: number | null;
  r2_finish: number | null;
  r3_finish: number | null;
  is_title: boolean;
  f1_elo: number | null;
  f2_elo: number | null;
  f1_adj_elo?: number | null;
  f2_adj_elo?: number | null;
  f1_inj?: number | null;
  f2_inj?: number | null;
  f1_rust?: number | null;
  f2_rust?: number | null;
  f1_american?: number | null;
  f2_american?: number | null;
  f1_edge_pct?: number;
  f2_edge_pct?: number;
  odds_source?: "bestfightodds" | "historical_archive" | "unavailable";
  actual_winner?: string | null;
  actual_method?: string | null;
  actual_round?: number | null;
  was_correct?: boolean | null;
  notes?: string[];
};

export type EventSummary = {
  id: string;
  name: string;
  url: string;
  full_text?: string;
  date?: string | null;
};

export type PastEventSummary = {
  id: string;
  name: string;
  date: string | null;
  url: string;
  correct: number;
  total: number;
  accuracy: number;
};

export type EventPredictions = {
  id: string;
  name: string;
  date: string | null;
  url: string;
  fights: FightPrediction[];
  source: "cache" | "live";
};

export type EloHistoryPoint = {
  fightdate: string;
  elobeforefight: number;
  eloafterfight: number;
  elochange: number;
  result: string | null;
  method: string | null;
  opponenteloebeforefight?: number | null;
  expectedwinprob?: number | null;
};

export type RecentFight = {
  date: string;
  opponentname: string;
  result: string;
  winnername: string | null;
  method: string;
  methoddetail: string | null;
  round: number | null;
  time: string | null;
  eventname: string;
  istitlefight: boolean;
  ismainevent: boolean;
};

export type FighterProfile = {
  id: number;
  name: string;
  nickname: string | null;
  url: string | null;
  weight_class: string | null;
  stance: string | null;
  height: string | null;
  weight: string | null;
  reach: string | null;
  leg_reach: string | null;
  dob: string | null;
  age: number | null;
  place_of_birth: string | null;
  record: {
    wins: number | null;
    losses: number | null;
    draws: number | null;
    total_fights: number | null;
  };
  days_since_last_fight: number | null;
  is_active: boolean | null;
  career_stats: {
    slpm: number | null;
    str_acc: number | null;
    sapm: number | null;
    str_def: number | null;
    td_avg: number | null;
    td_acc: number | null;
    td_def: number | null;
    sub_avg: number | null;
    avg_fight_duration: number | null;
    first_round_finish_rate: number | null;
    decision_rate: number | null;
    elo_rating: number | null;
    peak_elo: number | null;
  };
  elo_history: EloHistoryPoint[];
  recent_fights: RecentFight[];
};

export type TierStat = { total: number; correct: number; accuracy: number };
export type PaperBucket = {
  bets: number;
  wagered: number;
  returned: number;
  profit: number;
  roi: number;
  // True once `bets` clears the min-sample threshold (min_roi_bets). Paper ROI
  // below that is statistical noise and should not be shown as a real figure.
  reportable: boolean;
};

export type PerformanceStats = {
  total: number;
  correct: number;
  accuracy: number;
  tier_stats: Record<string, TierStat>;
  bucket_stats: Record<string, TierStat>;
  source_stats: Record<string, TierStat>;
  event_stats: Record<string, { total: number; correct: number; accuracy: number; date: string }>;
  high_conf_total: number;
  high_conf_correct: number;
  high_conf_accuracy: number;
  paper_trading: Record<string, PaperBucket>;
  min_roi_bets: number;
  num_events: number;
};

export type PerformanceSummary = {
  prediction_type: string;
  counts: Record<string, { total: number; resolved: number; correct: number }>;
  stats: PerformanceStats | null;
  available_years?: number[];
  selected_year?: number | null;
};
