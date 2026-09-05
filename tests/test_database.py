import unittest, tempfile, sqlite3
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from sql import build_database as db

class DatabaseTests(unittest.TestCase):
    def test_reload_and_constraints(self):
        frame=pd.DataFrame({'Country Code':['USA'], 'Country Name Standardized':['United States'], 'Region':['North America'], 'Indicator Short':['GDP'],'Year':[2024], 'Value':[10.]})
        with tempfile.TemporaryDirectory() as folder, patch.object(db,'script_dir',folder):
            conn=db.create_database()
            db.populate_database(conn,frame);db.populate_database(conn,frame)
            self.assertEqual(conn.execute('SELECT count(*) FROM "values"').fetchone()[0],1)
            self.assertEqual(conn.execute('PRAGMA foreign_key_check').fetchall(),[])
            with self.assertRaises(sqlite3.IntegrityError):conn.execute('INSERT INTO "values" VALUES ("USA","GDP",2024,20)')
            with self.assertRaises(sqlite3.IntegrityError):conn.execute('INSERT INTO "values" VALUES ("XXX","GDP",2024,20)')
            conn.close()

if __name__=='__main__':unittest.main()
