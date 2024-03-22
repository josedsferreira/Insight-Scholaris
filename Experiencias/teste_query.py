import psycopg2
import hashlib
import bcrypt

# Get user input
user_id = 1
password = "1234"

# Hash the password
# Hash the password
password = password.encode()  # Convert to bytes
salt = bcrypt.gensalt()  # Generate a salt
hashed_password = bcrypt.hashpw(password, salt)  # Hash the password

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    dbname="is_db_prototype",
    user="postgres",
    password="postgrespw",
    host="localhost",
    port="5432"
)

# Create a cursor object
cur = conn.cursor()

# Create the UPDATE query
query = """
UPDATE users
SET password = %s
WHERE user_id = %s;
"""

# Execute the query
cur.execute(query, (hashed_password, user_id))

# Commit the changes
conn.commit()

# Close the cursor and connection
cur.close()
conn.close()