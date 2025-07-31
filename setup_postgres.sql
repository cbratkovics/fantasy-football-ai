-- Create user and database for Fantasy Football AI
CREATE USER fantasy_user WITH PASSWORD 'fantasy_pass';
CREATE DATABASE fantasy_football OWNER fantasy_user;
GRANT ALL PRIVILEGES ON DATABASE fantasy_football TO fantasy_user;

-- Connect to the database
\c fantasy_football;

-- Grant schema permissions
GRANT ALL ON SCHEMA public TO fantasy_user;