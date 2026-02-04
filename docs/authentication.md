# Authentication

PyHellen supports optional token-based authentication to secure your API.

## Overview

- **Disabled by default** - API is publicly accessible
- **Token-based** - Bearer tokens or API keys
- **Scope-based permissions** - read, write, admin
- **SQLite storage** - Tokens stored in local database

## Setup

### 1. Generate a Secret Key

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. Configure Environment

Edit your `.env` file:

```bash
AUTH_ENABLED=true
SECRET_KEY="your-generated-secret-key-here"
TOKEN_DB_PATH="tokens.db"
AUTO_CREATE_ADMIN_TOKEN=true
```

### 3. Start the Server

On first run with `AUTO_CREATE_ADMIN_TOKEN=true`, an admin token is created and logged:

```
INFO: Created initial admin token: phk_xxxxxxxxxxxxxxxx
INFO: Store this token securely - it will not be shown again!
```

**Save this token!** It cannot be retrieved later.

## Using Tokens

### Authorization Header (Bearer)

```bash
curl -H "Authorization: Bearer phk_your_token_here" \
  "http://localhost:8000/api/tag/lasla?text=hello"
```

### X-API-Key Header

```bash
curl -H "X-API-Key: phk_your_token_here" \
  "http://localhost:8000/api/tag/lasla?text=hello"
```

## Token Scopes

| Scope | Permissions |
|-------|-------------|
| `read` | Use tagging endpoints (GET/POST /api/tag, /api/batch, /api/stream, /api/models, /api/cache/stats) |
| `write` | All `read` permissions + modify cache, load/unload models |
| `admin` | Full access including token management and model activation/deactivation |

## Admin Endpoints

All admin endpoints require `admin` scope.

### Authentication Status

```http
GET /admin/auth/status
```

```json
{
  "authenticated": true,
  "auth_enabled": true,
  "token_name": "admin",
  "scopes": ["admin"]
}
```

### List Tokens

```http
GET /admin/tokens
```

```json
[
  {
    "id": 1,
    "name": "admin",
    "scopes": ["admin"],
    "created_at": "2024-01-15T10:00:00",
    "expires_at": null,
    "last_used_at": "2024-01-15T12:00:00",
    "is_active": true
  }
]
```

### Create Token

```http
POST /admin/tokens
Content-Type: application/json

{
  "name": "my-app",
  "scopes": ["read", "write"],
  "expires_days": 30
}
```

**Response:**
```json
{
  "id": 2,
  "name": "my-app",
  "token": "phk_xxxxxxxxxxxxxxxx",
  "scopes": ["read", "write"],
  "created_at": "2024-01-15T10:00:00",
  "expires_at": "2024-02-14T10:00:00"
}
```

**Important:** The `token` field is only returned once. Store it securely!

### Revoke Token

```http
DELETE /admin/tokens/{token_id}
```

Revokes a token (soft delete). The token can no longer be used.

### Delete Token Permanently

```http
DELETE /admin/tokens/{token_id}/permanent
```

Permanently removes a token from the database.

### Cleanup Expired Tokens

```http
POST /admin/tokens/cleanup
```

Removes all expired tokens.

### Token Statistics

```http
GET /admin/tokens/stats
```

```json
{
  "total": 5,
  "active": 4,
  "expired": 1,
  "auth_enabled": true
}
```

## Model Management (Admin)

### List All Models

```http
GET /admin/models?include_inactive=true
```

### Activate Model

```http
POST /admin/models/{code}/activate
```

### Deactivate Model

```http
POST /admin/models/{code}/deactivate
```

Deactivated models cannot be used for tagging and are unloaded from memory.

## Audit Logs

All admin actions are logged for security.

### View Audit Log

```http
GET /admin/audit?limit=100&offset=0
```

### Audit Statistics

```http
GET /admin/audit/stats?hours=24
```

### Cleanup Old Entries

```http
POST /admin/audit/cleanup?days=90
```

## Request Logs

Track all API requests.

### View Request Log

```http
GET /admin/requests?limit=100&model=lasla
```

### Request Statistics

```http
GET /admin/requests/stats?hours=24
```

### View Errors

```http
GET /admin/requests/errors?limit=100
```

## Security Best Practices

1. **Always use HTTPS in production**
2. **Generate a strong SECRET_KEY** (at least 32 characters)
3. **Never commit SECRET_KEY to version control**
4. **Use minimal scopes** - only grant permissions needed
5. **Set token expiration** for temporary access
6. **Review audit logs regularly**
7. **Cleanup expired tokens periodically**
