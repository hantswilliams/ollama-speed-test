import Database from "better-sqlite3";
import path from "node:path";

const DB_PATH = path.resolve(process.cwd(), "..", "data", "usage.db");

let _db: Database.Database | null = null;

export function db(): Database.Database {
  if (_db) return _db;
  _db = new Database(DB_PATH, { readonly: true, fileMustExist: false });
  _db.pragma("journal_mode = WAL");
  return _db;
}

export type RequestRow = {
  id: number;
  timestamp: string;
  model: string;
  endpoint: string;
  streamed: number;
  prompt_tokens: number | null;
  output_tokens: number | null;
  prompt_eval_duration_ns: number | null;
  eval_duration_ns: number | null;
  load_duration_ns: number | null;
  total_duration_ns: number | null;
  output_tps: number | null;
  prompt_tps: number | null;
  wall_time_sec: number | null;
  client_ip: string | null;
};

export function tableExists(): boolean {
  try {
    const row = db()
      .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='requests'")
      .get();
    return !!row;
  } catch {
    return false;
  }
}
