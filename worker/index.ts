import { Container, getContainer } from "@cloudflare/containers";

export class LedgerlineApp extends Container {
  defaultPort = 8000;      // uvicorn port from the Dockerfile
  sleepAfter = "30m";      // stops billing when idle; wakes on next request
}

export default {
  async fetch(request: Request, env: { LEDGERLINE: DurableObjectNamespace<LedgerlineApp> }) {
    return getContainer(env.LEDGERLINE).fetch(request);
  },
};
