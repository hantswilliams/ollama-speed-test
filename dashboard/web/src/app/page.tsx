import TotalsPanel from "@/components/TotalsPanel";
import LoadedModelsPanel from "@/components/LoadedModelsPanel";
import TpsChart from "@/components/TpsChart";
import RequestsTable from "@/components/RequestsTable";

export default function Page() {
  return (
    <div className="app">
      <header>
        <div>
          <h1>Ollama Dashboard</h1>
          <div className="sub">
            Logging proxy on <code>:11435</code> &rarr; Ollama on <code>:11434</code>
          </div>
        </div>
        <div className="sub">auto-refreshes every 3s</div>
      </header>

      <div className="grid">
        <div className="col-12">
          <TotalsPanel />
        </div>

        <div className="col-8">
          <div className="card">
            <h2>Output tok/s (recent requests)</h2>
            <TpsChart />
          </div>
        </div>

        <div className="col-4">
          <div className="card">
            <h2>Loaded models</h2>
            <LoadedModelsPanel />
          </div>
        </div>

        <div className="col-12">
          <div className="card">
            <h2>Recent requests</h2>
            <RequestsTable />
          </div>
        </div>
      </div>
    </div>
  );
}
