import psycopg2

def migrate():
    conn_str = "postgresql://admin:password123@localhost:5432/neuro_db"
    try:
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()
        
        columns = [
            ("grade", "VARCHAR"),
            ("prior_treatment", "VARCHAR"),
            ("idh_mutation", "VARCHAR"),
            ("mgmt_methylation", "VARCHAR")
        ]
        
        for col_name, col_type in columns:
            print(f"Adding column {col_name} to clinical_data...")
            try:
                cur.execute(f"ALTER TABLE clinical_data ADD COLUMN {col_name} {col_type};")
                print(f" -> Added {col_name}")
            except psycopg2.errors.DuplicateColumn:
                conn.rollback()
                print(f" -> Column {col_name} already exists, skipping.")
                continue
            except Exception as e:
                print(f" -> Error adding {col_name}: {e}")
                conn.rollback()
                continue
            
        conn.commit()
        cur.close()
        conn.close()
        print("Migration completed!")
        
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    migrate()
