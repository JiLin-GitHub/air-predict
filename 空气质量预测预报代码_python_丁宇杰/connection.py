import cx_Oracle


class DBConnection(object):
    def __init__(self, ip, port, serviceName, username, password):
        self.__ip = ip
        self.__port = port
        self.__serviceName = serviceName
        self.__username = username
        self.__password = password

    def getConnection(self):
        dsnStr = cx_Oracle.makedsn(self.__ip, self.__port, self.__serviceName)
        connection = cx_Oracle.connect(user=self.__username, password=self.__password, dsn=dsnStr)
        return connection

    @property
    def connection(self):
        return self._connect

    @connection.setter
    def connection(self, value):
        self._connect = value

    def closeConnection(self):
        self._connect.close()

    def getCorsor(self):
        return self._connect.cursor()

    @property
    def cursor(self):
        return self._cursor

    @cursor.setter
    def cursor(self, value):
        self._cursor = value

    def closeCursor(self):
        self._cursor.close()

    def query(self, sql):
        cursor = self.getCorsor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result

    def insert(self, sql):
        cursor = self.getCorsor()
        cursor.executemany(sql)
        cursor.close()
        self._connect.commit()

    def DDLDB_P(self, sql, para):
        cursor = self.getCorsor()
        cursor.execute(sql, para)
        cursor.close()
        self._connect.commit()
