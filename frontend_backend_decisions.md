# Frontend / Backend Decision Lock

This document records the agreed implementation decisions for the web application layer of this project.

## Purpose

The existing repository already covers the ML-serving core:
- model training
- MLflow tracking and registration
- a raw prediction API

The next implementation phase is the **web application layer** used for the live demo.

## Locked Scope

We are building:
- a browser-based frontend
- an application backend for auth, validation, status, and ML-service integration
- a protected demo flow around the existing ML inference service

We are **not** redesigning the model, retraining pipeline, or MLflow internals as part of this task.

## Architecture Decisions

### High-level architecture
- Keep the current ML inference API as a **separate service**.
- Add a new **application backend** for web app behavior.
- Add a new **frontend** as a separate app.
- The application backend calls the ML service over **internal HTTP**.
- The frontend calls the application backend, not the ML service directly.

### Deployment shape
- Frontend stack: **React + Vite + TypeScript**.
- Backend stack: **FastAPI**.
- Database: existing **PostgreSQL service**.
- Auth: **password hash in database + signed session cookie**.
- Final demo deployment should be **same-origin** for the browser.
- The built frontend should be served by the application backend.

### Repository structure target
- Keep current ML service code separate.
- Add a new `backend/` area for the application backend.
- Add a new `frontend/` area for the React app.

## Frontend Decisions

### Required pages
- `/signup`
- `/login`
- `/demo`
- `/status`

### General UI decisions
- No public landing page.
- English UI is acceptable.
- Main prediction flow should happen on a **single protected page**.
- Prediction results should render on the **same page** as the input form.

### Demo page behavior
The main `/demo` page should include:
- title input
- article body input
- submit action
- predicted tags display

### Prediction output
- Show **tags only**.
- Do **not** show confidence scores in the user-facing UI.
- Support **one article at a time**, not batch prediction.

## Authentication Decisions

### Account model
- Real **sign-up** is included.
- User identity is based on:
  - username
  - password
- No email is required.
- After successful sign-up, the user should be **logged in immediately**.
- Access to protected pages is available immediately after sign-up.

### Auth mechanism
- Store **hashed passwords** in the database.
- Use **signed session cookie auth**.
- Do not add a server-side session table for this version.

### Credential policy
- Keep validation minimal.
- Username must be unique.
- No minimum password length requirement for this demo.
- No advanced password complexity rules.
- No role system or admin system for this version.

## Backend Decisions

### Application backend responsibilities
The new backend should own:
- sign-up
- login
- logout
- protected access control
- input validation
- clean error responses
- prediction proxying to the ML service
- health endpoint
- status/monitoring endpoints

### Health and degraded mode
- Expose a **public health endpoint**.
- The application backend should still **start in degraded mode** if the ML service is unavailable.
- If the ML service is down, the UI should show a **friendly error**.
- Status/health should clearly reflect backend, database, and ML-service state.

## Status / Monitoring Decisions

### Status page
- Include a separate protected `/status` page.
- All authenticated users may view `/status`.
- No admin-only status view for now.

### What `/status` should show
- global system metrics
- current user's own recent request history
- health information
- model/service availability information

### Monitoring data policy
For request history, store and show only a **short excerpt/preview** of submitted content, not the full article body.

A suitable status view should support values such as:
- total request count
- failed request count
- average response time
- recent user requests
- success/failure state
- model or ML service status

## Data and Persistence Decisions

### PostgreSQL usage
- Reuse the existing PostgreSQL service.
- Use a **separate application database** inside that same Postgres service.
- Keep app data separate from MLflow metadata.

### App-owned persistent data
Store app-layer data in PostgreSQL, including:
- users
- prediction activity / monitoring records

### Schema management
- Auto-create app tables at backend startup.
- Do not add a migration framework for this version.

## Testing Decisions

### Required testing scope
- Implement **backend tests only** for this phase.
- Frontend will be validated through the live app flow rather than a formal frontend test suite.

## Requirement Interpretation Notes

Based on the current project report and requirement summary:
- a browser-based frontend is required
- basic authentication is required
- backend integration with ML inference is required
- status/monitoring support is required
- end-to-end deployment is required

The current interpretation is that **local Docker Compose deployment is sufficient** as long as the full system works end to end during the demonstration.

## Summary

The agreed implementation target is:
- **React + Vite + TypeScript frontend**
- **FastAPI application backend**
- **existing ML service kept separate**
- **PostgreSQL-backed users and monitoring data**
- **session-cookie auth**
- **protected demo flow and status page**

This document is the locked decision baseline for the next implementation task.