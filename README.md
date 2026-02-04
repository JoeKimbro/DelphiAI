# DelphiAI
This is a Sports Betting stat analyzer that can help you predict the best course of action for your money!

## Quick Start

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the database
```bash
docker-compose up -d
```

### 3. Access pgAdmin
- URL: http://localhost:5050
- Email: `admin@delphi.local`
- Password: `admin123`

### 4. Connect to PostgreSQL in pgAdmin
- Host: `postgres`
- Port: `5432`
- Database: `delphi_db`
- Username: `delphi_user`
- Password: `delphi_password`

## Docker Commands

| Command | Description |
|---------|-------------|
| `docker-compose up -d` | Start all containers |
| `docker-compose down` | Stop all containers |
| `docker-compose restart` | Restart all containers |
| `docker-compose logs -f` | View live logs |
| `docker-compose logs postgres` | View database logs |
| `docker-compose ps` | List running containers |

## Database Commands

| Command | Description |
|---------|-------------|
| `docker-compose down -v` | Stop containers and delete all data |
| `docker-compose exec postgres psql -U delphi_user -d delphi_db` | Open PostgreSQL shell |

## Troubleshooting

**Reset the database (delete all data):**
```bash
docker-compose down -v
docker-compose up -d
```

**Check if containers are running:**
```bash
docker-compose ps
```

**View container logs:**
```bash
docker-compose logs -f
```
