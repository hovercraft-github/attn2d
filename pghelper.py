import psycopg2
#import logging

class PgHelper(object):

    def __init__(self):
        self.pg_object = None

    def __enter__(self):
        class pgObject:
            def __init__(self):
                #self.logger = logging.getLogger(jobname)
                self.connection = psycopg2.connect(user="postgres",
                                        password="",
                                        host="algodb",
                                        port="5432",
                                        database="algodb")
                self.cursor = None

            def __iter__(self):
                return self

            def __next__(self):
                if self.cursor == None:
                    return None
                row = self.cursor.__next__()
                return { description[0]: row[col] for col, description in enumerate(self.cursor.description) }

            def cleanup(self):
                if not self.cursor == None:
                    self.cursor.close()
                    self.cursor = None
                if not self.connection == None:
                    self.connection.close()
                    self.connection = None

            def yield_all(self, cursor):
                while True:
                    if cursor.description is None:
                        # No recordset for INSERT, UPDATE, CREATE, etc
                        pass
                    else:
                        # Recordset for SELECT, yield data
                        yield cursor.fetchall()
                        # Or yield column names with
                        # yield [col[0] for col in cursor.description]

                    # Go to the next recordset
                    if not cursor.nextset():
                        # End of recordsets
                        return

            def exec(self, SQL, bind_vars=None):
                if not self.cursor == None:
                    self.cursor.close()
                    self.cursor = None
                self.cursor = self.connection.cursor()
                self.cursor.execute(SQL, bind_vars)
                return self.cursor

        self.pg_object = pgObject()
        return self.pg_object

    def __exit__(self, exc_type, exc_value, traceback):
        self.pg_object.cleanup()
        return False
