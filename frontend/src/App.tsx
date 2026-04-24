import { FormEvent, ReactNode, createContext, useContext, useEffect, useMemo, useState } from "react";
import { Navigate, NavLink, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import logo from "./logo.png";
import {
  ApiError,
  DashboardData,
  SessionUser,
  getDashboardData,
  getSession,
  login as apiLogin,
  logout as apiLogout,
  predictTags,
  signup as apiSignup,
} from "./api";

type AuthContextValue = {
  user: SessionUser | null;
  ready: boolean;
  busy: boolean;
  signIn: (username: string, password: string) => Promise<void>;
  signUp: (username: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  refresh: () => Promise<void>;
};

type DemoState = {
  tags: string[];
  note?: string;
  unavailable?: boolean;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export default function App() {
  const auth = useProvideAuth();

  return (
    <AuthContext.Provider value={auth}>
      <AppFrame>
        <Routes>
          <Route path="/" element={<HomeRedirect />} />
          <Route
            path="/signup"
            element={
              <GuestOnlyRoute>
                <AuthPage mode="signup" />
              </GuestOnlyRoute>
            }
          />
          <Route
            path="/login"
            element={
              <GuestOnlyRoute>
                <AuthPage mode="login" />
              </GuestOnlyRoute>
            }
          />
          <Route
            path="/demo"
            element={
              <ProtectedRoute>
                <DemoPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/status"
            element={
              <ProtectedRoute>
                <StatusPage />
              </ProtectedRoute>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AppFrame>
    </AuthContext.Provider>
  );
}

function useProvideAuth(): AuthContextValue {
  const [user, setUser] = useState<SessionUser | null>(null);
  const [ready, setReady] = useState(false);
  const [busy, setBusy] = useState(false);

  const refresh = async () => {
    try {
      setUser(await getSession());
    } finally {
      setReady(true);
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const signIn = async (username: string, password: string) => {
    setBusy(true);
    try {
      setUser(await apiLogin(username, password));
      setReady(true);
    } finally {
      setBusy(false);
    }
  };

  const signUp = async (username: string, password: string) => {
    setBusy(true);
    try {
      setUser(await apiSignup(username, password));
      setReady(true);
    } finally {
      setBusy(false);
    }
  };

  const signOut = async () => {
    setBusy(true);
    try {
      await apiLogout();
      setUser(null);
      setReady(true);
    } finally {
      setBusy(false);
    }
  };

  return useMemo(
    () => ({ user, ready, busy, signIn, signUp, signOut, refresh }),
    [user, ready, busy],
  );
}

function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("Auth context is unavailable.");
  }
  return context;
}

function AppFrame({ children }: { children: ReactNode }) {
  const { user, busy, signOut } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const isAuthRoute = location.pathname === "/login" || location.pathname === "/signup";

  const handleLogout = async () => {
    try {
      await signOut();
      navigate("/login", { replace: true });
    } catch {
      navigate("/login", { replace: true });
    }
  };

  if (isAuthRoute) {
    return (
      <div className="app-shell auth-shell">
        <main className="auth-page-wrap">{children}</main>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <NavLink to={user ? "/demo" : "/login"} className="brand-link" aria-label="Tag Plus home">
          <img src={logo} alt="Tag Plus" className="brand-logo" />
          <span className="brand-wordmark">Tag Plus</span>
        </NavLink>

        <nav className="topbar-nav" aria-label="Primary">
          {user ? (
            <>
              <NavLink to="/demo" className={topbarLinkClass}>
                Demo
              </NavLink>
              <NavLink to="/status" className={topbarLinkClass}>
                Status
              </NavLink>
              <span className="user-chip">{user.username}</span>
              <button type="button" className="button button-subtle" onClick={handleLogout} disabled={busy}>
                {busy ? "Working..." : "Log out"}
              </button>
            </>
          ) : null}
        </nav>
      </header>

      <main className="page-wrap">{children}</main>
    </div>
  );
}

function HomeRedirect() {
  const { user, ready } = useAuth();

  if (!ready) {
    return <LoadingPanel label="One moment" detail="Getting things ready for you." />;
  }

  return <Navigate to={user ? "/demo" : "/login"} replace />;
}

function ProtectedRoute({ children }: { children: ReactNode }) {
  const { user, ready } = useAuth();
  const location = useLocation();

  if (!ready) {
    return <LoadingPanel label="Loading" detail="Setting up your workspace." />;
  }

  if (!user) {
    return <Navigate to="/login" replace state={{ from: `${location.pathname}${location.search}` }} />;
  }

  return <>{children}</>;
}

function GuestOnlyRoute({ children }: { children: ReactNode }) {
  const { user, ready } = useAuth();

  if (!ready) {
    return <LoadingPanel label="Loading" detail="Just a moment." />;
  }

  if (user) {
    return <Navigate to="/demo" replace />;
  }

  return <>{children}</>;
}

function AuthPage({ mode }: { mode: "login" | "signup" }) {
  const { signIn, signUp, busy } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isSignup = mode === "signup";
  const redirectState = location.state as { from?: string } | null;
  const redirectTo = typeof redirectState?.from === "string" ? redirectState.from : "/demo";
  const passwordInputType = showPassword ? "text" : "password";
  const passwordToggleLabel = showPassword ? "Hide password" : "Show password";

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);

    const trimmedUsername = username.trim();
    if (!trimmedUsername || !password) {
      setError("Enter both a username and password.");
      return;
    }

    try {
      if (isSignup) {
        await signUp(trimmedUsername, password);
      } else {
        await signIn(trimmedUsername, password);
      }
      navigate(redirectTo, { replace: true });
    } catch (submitError) {
      setError(getErrorMessage(submitError, isSignup ? "Unable to create your account." : "Unable to sign you in."));
    }
  };

  if (isSignup) {
    return (
      <section className="signup-layout">
        <aside className="signup-aside" aria-hidden="true">
          <div className="signup-aside-inner">
            <h1>Tag Plus</h1>
            <p>Tag your articles instantly. Organized, accurate, effortless.</p>
          </div>
        </aside>

        <main className="signup-main">
          <article className="auth-card auth-card-signup">
            <header className="auth-card-head">
              <h2>Create Account</h2>
              <p>Sign up to get started</p>
            </header>

            <form className="auth-form" onSubmit={handleSubmit} autoComplete="off">
              <label className="field" htmlFor="signup-username">
                <span>Username</span>
                <input
                  id="signup-username"
                  className="input"
                  autoComplete="off"
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  placeholder="Enter your username"
                />
              </label>

              <label className="field" htmlFor="signup-password">
                <span>Password</span>
                <div className="password-field">
                  <input
                    id="signup-password"
                    className="input password-input"
                    type={passwordInputType}
                    autoComplete="off"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    placeholder="Create a password"
                  />
                  <button
                    type="button"
                    className="password-toggle"
                    aria-label={passwordToggleLabel}
                    aria-pressed={showPassword}
                    onClick={() => setShowPassword((current) => !current)}
                  >
                    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                      {showPassword ? (
                        <path
                          d="M3 5.27 4.28 4 20 19.72 18.73 21l-3.04-3.04A10.78 10.78 0 0 1 12 18c-5 0-9.27-3.11-11-6 1.02-1.71 3.06-3.69 5.67-4.8L3 5.27ZM12 8a4 4 0 0 1 4 4c0 .73-.2 1.41-.55 1.99l-1.58-1.58c.08-.13.13-.27.13-.41a2 2 0 0 0-2-2c-.14 0-.28.05-.41.13L10.01 8.55A3.96 3.96 0 0 1 12 8Zm0-2c5 0 9.27 3.11 11 6-.67 1.12-1.73 2.35-3.11 3.38l-1.43-1.43A9.69 9.69 0 0 0 20.62 12C19.08 9.84 15.8 8 12 8c-.74 0-1.47.07-2.16.2L8.17 6.53C9.38 6.19 10.67 6 12 6Z"
                          fill="currentColor"
                        />
                      ) : (
                        <path
                          d="M12 5c5 0 9.27 3.11 11 7-1.73 3.89-6 7-11 7S2.73 15.89 1 12c1.73-3.89 6-7 11-7Zm0 2C8.2 7 4.92 8.84 3.38 12 4.92 15.16 8.2 17 12 17s7.08-1.84 8.62-5C19.08 8.84 15.8 7 12 7Zm0 2.5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5Z"
                          fill="currentColor"
                        />
                      )}
                    </svg>
                  </button>
                </div>
              </label>

              {error ? <div className="notice notice-error">{error}</div> : null}

              <button type="submit" className="button button-block" disabled={busy}>
                {busy ? "Please wait..." : "Create Account"}
              </button>
            </form>

            <footer className="auth-footer">
              <span>Already have an account?</span>
              <NavLink className="text-link" to="/login">
                Sign In
              </NavLink>
            </footer>
          </article>
        </main>
      </section>
    );
  }

  return (
    <section className="login-layout">
      <header className="login-brand">
        <img src={logo} alt="Tag Plus" className="login-logo" />
      </header>

      <article className="auth-card auth-card-login">
        <form className="auth-form" onSubmit={handleSubmit} autoComplete="off">
          <label className="field" htmlFor="login-username">
            <span>Username</span>
            <input
              id="login-username"
              className="input"
              autoComplete="off"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              placeholder="Enter your username"
            />
          </label>

          <label className="field" htmlFor="login-password">
            <span>Password</span>
            <div className="password-field">
              <input
                id="login-password"
                className="input password-input"
                type={passwordInputType}
                autoComplete="off"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="Enter your password"
              />
              <button
                type="button"
                className="password-toggle"
                aria-label={passwordToggleLabel}
                aria-pressed={showPassword}
                onClick={() => setShowPassword((current) => !current)}
              >
                <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                  {showPassword ? (
                    <path
                      d="M3 5.27 4.28 4 20 19.72 18.73 21l-3.04-3.04A10.78 10.78 0 0 1 12 18c-5 0-9.27-3.11-11-6 1.02-1.71 3.06-3.69 5.67-4.8L3 5.27ZM12 8a4 4 0 0 1 4 4c0 .73-.2 1.41-.55 1.99l-1.58-1.58c.08-.13.13-.27.13-.41a2 2 0 0 0-2-2c-.14 0-.28.05-.41.13L10.01 8.55A3.96 3.96 0 0 1 12 8Zm0-2c5 0 9.27 3.11 11 6-.67 1.12-1.73 2.35-3.11 3.38l-1.43-1.43A9.69 9.69 0 0 0 20.62 12C19.08 9.84 15.8 8 12 8c-.74 0-1.47.07-2.16.2L8.17 6.53C9.38 6.19 10.67 6 12 6Z"
                      fill="currentColor"
                    />
                  ) : (
                    <path
                      d="M12 5c5 0 9.27 3.11 11 7-1.73 3.89-6 7-11 7S2.73 15.89 1 12c1.73-3.89 6-7 11-7Zm0 2C8.2 7 4.92 8.84 3.38 12 4.92 15.16 8.2 17 12 17s7.08-1.84 8.62-5C19.08 8.84 15.8 7 12 7Zm0 2.5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5Z"
                      fill="currentColor"
                    />
                  )}
                </svg>
              </button>
            </div>
          </label>

          {error ? <div className="notice notice-error">{error}</div> : null}

          <div className="auth-actions">
            <button type="submit" className="button button-block" disabled={busy}>
              {busy ? "Please wait..." : "Sign In"}
            </button>
            <NavLink className="auth-secondary-link" to="/signup">
              Sign up
            </NavLink>
          </div>
        </form>
      </article>
    </section>
  );
}

function DemoPage() {
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DemoState | null>(null);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const clearForm = () => {
    setTitle("");
    setBody("");
    setError(null);
    setResult(null);
  };

  const copyTag = async (tag: string, index: number) => {
    try {
      await navigator.clipboard.writeText(tag);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 1500);
    } catch {
      // clipboard not available
    }
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    setResult(null);

    if (!body.trim()) {
      setError("Paste article text before requesting tags.");
      return;
    }

    setSubmitting(true);
    try {
      const response = await predictTags({ title, body });
      setResult({
        tags: response.tags,
        note:
          response.tags.length === 0
            ? response.message || "We could not find any strong tag suggestions for this article."
            : response.message,
      });
    } catch (submitError) {
      setError(getErrorMessage(submitError, "Unable to fetch predicted tags."));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section className="demo-layout">
      <header className="section-heading">
        <h1>Article Analysis</h1>
        <p>Paste your article and we'll suggest the best tags for it.</p>
      </header>

      <article className="surface-card composer-card">
        <form className="composer-form" onSubmit={handleSubmit}>
          <label className="field field-secondary" htmlFor="article-title">
            <span>
              Title <em>(Optional)</em>
            </span>
            <input
              id="article-title"
              className="input input-subtle"
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              placeholder="Add a title for context..."
            />
          </label>

          <div className="composer-main-field">
            <div className="composer-field-header">
              <label className="composer-label" htmlFor="article-body">
                Paste Article
              </label>
              <button type="button" className="clear-link" onClick={clearForm} disabled={submitting}>
                Clear All
              </button>
            </div>

            <textarea
              id="article-body"
              className="input textarea"
              value={body}
              onChange={(event) => setBody(event.target.value)}
              placeholder="Paste the full text of your article here..."
            />
          </div>

          {error ? <div className="notice notice-error">{error}</div> : null}

          <div className="composer-actions">
            <button type="submit" className="button" disabled={submitting}>
              {submitting ? "Analyzing..." : "Submit for Analysis"}
            </button>
          </div>
        </form>
      </article>

      <section className="results-section">
        <div className="results-heading-row">
          <h2>Predicted Tags</h2>
          {submitting ? (
            <span className="result-state result-state-analyzing">Analyzing…</span>
          ) : result ? (
            result.unavailable ? (
              <span className="result-state result-state-warning">Temporarily Limited</span>
            ) : result.tags.length > 0 ? (
              <span className="result-state result-state-success">Analysis Complete</span>
            ) : null
          ) : null}
        </div>

        {submitting ? (
          <div className="analyzing-panel">
            <div className="analyzing-shimmer" />
            <div className="analyzing-shimmer analyzing-shimmer-short" />
            <div className="analyzing-shimmer analyzing-shimmer-shorter" />
            <p>Reading your article and finding the best tags…</p>
          </div>
        ) : !result ? (
          <EmptyState title="Ready for analysis" detail="Tags will appear here after analysis." />
        ) : result.unavailable ? (
          <EmptyState
            title="Service unavailable"
            detail={result.note ?? "The service is temporarily unavailable."}
            tone="warning"
          />
        ) : result.tags.length > 0 ? (
          <div className="results-card">
            <div className="result-list">
              {result.tags.map((tag, index) => (
                <button
                  type="button"
                  key={`${tag}-${index}`}
                  className={`result-row result-row-rank-${Math.min(index, 4)}${index === 0 ? " result-row-primary" : ""}${
                    copiedIndex === index ? " result-row-copied" : ""
                  }`}
                  onClick={() => void copyTag(tag, index)}
                  title="Click to copy"
                >
                  <span className="result-row-rank">#{index + 1}</span>
                  <span className="result-row-label">{tag}</span>
                  <span className="result-row-action">
                    {copiedIndex === index ? "Copied!" : "Copy"}
                  </span>
                </button>
              ))}
            </div>
            {result.note ? <p className="result-note">{result.note}</p> : null}
          </div>
        ) : (
          <EmptyState
            title="Analysis complete"
            detail={result.note ?? "We could not find any strong tag suggestions for this article."}
            tone="result"
          />
        )}
      </section>
    </section>
  );
}

function StatusPage() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadStatus = async () => {
    setLoading(true);
    setError(null);

    try {
      setData(await getDashboardData());
    } catch (statusError) {
      setError(getErrorMessage(statusError, "Unable to load system status."));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadStatus();
  }, []);

  return (
    <section className="status-page">
      <header className="status-heading">
        <div>
          <h1>System Overview</h1>
          <p>See how things are running and review your recent activity.</p>
        </div>
        <button type="button" className="button button-subtle" onClick={() => void loadStatus()} disabled={loading}>
          {loading ? "Refreshing..." : "Refresh"}
        </button>
      </header>

      {error ? <div className="notice notice-error">{error}</div> : null}

      <section className="metrics-grid">
        <MetricCard label="Total Requests" value={formatMetric(data?.totalRequests)} />
        <MetricCard label="Failed Requests" value={formatMetric(data?.failedRequests)} />
        <MetricCard label="Avg. Latency" value={formatLatency(data?.averageResponseTimeMs)} />
        <MetricCard label="Success Rate" value={formatSuccessRate(data?.successRate, data)} />
      </section>

      <section className="status-section surface-card">
        <div className="section-subheading">
          <h2>Service Health</h2>
        </div>

        {loading && !data ? (
          <LoadingInline label="Checking service status..." />
        ) : (
          <div className="service-health-grid">
            {(data?.services ?? []).map((service) => (
              <article key={service.label} className="service-health-item">
                <div className={`service-dot service-dot-${service.state}`} aria-hidden="true" />
                <div className="service-health-copy">
                  <strong>{service.label}</strong>
                  <p>{service.detail || "No additional details available."}</p>
                </div>
                <span className={`service-health-state service-health-state-${service.state}`}>{service.state}</span>
              </article>
            ))}
          </div>
        )}
      </section>

      <section className="activity-section">
        <div className="section-subheading">
          <h2>Recent Activity</h2>
        </div>

        <div className="activity-list">
          <div className="activity-list-head">
            <span>Time</span>
            <span>Title</span>
            <span>Content</span>
            <span>Tags</span>
            <span className="activity-cell-right">Latency</span>
            <span className="activity-cell-right">Status</span>
          </div>

          {loading && !data ? (
            <div className="surface-card activity-empty">
              <LoadingInline label="Loading recent activity..." />
            </div>
          ) : data?.recentRequests?.length ? (
            data.recentRequests.map((request, index) => {
              const tags = request.tags ?? [];
              const requestState = mapRequestState(request.status);

              return (
                <article key={request.id ?? `${request.preview}-${index}`} className="activity-row surface-card">
                  <div className="activity-cell activity-time">{formatTimestamp(request.createdAt)}</div>
                  <div className="activity-cell activity-title">{request.title || "—"}</div>
                  <div className="activity-cell activity-preview">{request.preview}</div>
                  <div className="activity-cell activity-tags">
                    {tags.length ? (
                      <div className="tag-row">
                        {tags.map((tag, tagIndex) => (
                          <span key={`${tag}-${tagIndex}`} className="tag-pill">
                            {tag}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span className="activity-muted">—</span>
                    )}
                  </div>
                  <div className="activity-cell activity-latency activity-cell-right">
                    {request.responseTimeMs !== undefined ? `${Math.round(request.responseTimeMs)}ms` : "—"}
                  </div>
                  <div className="activity-cell activity-status activity-cell-right">
                    <span className={`status-badge status-${requestState}`}>{request.status}</span>
                  </div>
                </article>
              );
            })
          ) : (
            <div className="surface-card activity-empty">
              <EmptyState title="No recent activity yet." />
            </div>
          )}
        </div>
      </section>
    </section>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <article className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
    </article>
  );
}

function EmptyState({
  title,
  detail,
  tone = "default",
}: {
  title: string;
  detail?: string;
  tone?: "default" | "warning" | "result";
}) {
  return (
    <div
      className={`empty-state${
        tone === "warning" ? " empty-state-warning" : tone === "result" ? " empty-state-result" : ""
      }`}
    >
      <strong>{title}</strong>
      {detail ? <p>{detail}</p> : null}
    </div>
  );
}

function LoadingPanel({ label, detail }: { label: string; detail: string }) {
  return (
    <section className="loading-panel surface-card">
      <div className="loading-dot" />
      <h2>{label}</h2>
      <p>{detail}</p>
    </section>
  );
}

function LoadingInline({ label }: { label: string }) {
  return (
    <div className="loading-inline">
      <div className="loading-dot" />
      <span>{label}</span>
    </div>
  );
}

function getErrorMessage(error: unknown, fallback: string) {
  if (error instanceof ApiError) {
    return error.message || fallback;
  }
  if (error instanceof Error) {
    return error.message || fallback;
  }
  return fallback;
}

function formatMetric(value?: number) {
  return value === undefined ? "—" : Intl.NumberFormat("en-US").format(value);
}

function formatLatency(value?: number) {
  if (value === undefined) {
    return "—";
  }
  return `${Math.round(value)} ms`;
}

function formatSuccessRate(successRate: number | undefined, data: DashboardData | null) {
  if (successRate !== undefined) {
    return `${Math.round(successRate * (successRate <= 1 ? 100 : 1))}%`;
  }

  if (data?.totalRequests && data.failedRequests !== undefined) {
    const succeeded = Math.max(data.totalRequests - data.failedRequests, 0);
    return `${Math.round((succeeded / data.totalRequests) * 100)}%`;
  }

  return "—";
}

function formatTimestamp(value?: string) {
  if (!value) {
    return "Time unavailable";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function mapRequestState(status: string) {
  const normalized = status.trim().toLowerCase();
  if (["success", "ok", "completed"].includes(normalized)) {
    return "healthy";
  }
  if (["degraded", "partial"].includes(normalized)) {
    return "degraded";
  }
  if (["failed", "error", "unavailable"].includes(normalized)) {
    return "unavailable";
  }
  return "unknown";
}

function topbarLinkClass({ isActive }: { isActive: boolean }) {
  return `topbar-link${isActive ? " topbar-link-active" : ""}`;
}
