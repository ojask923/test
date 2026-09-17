import sqlite3
conn = sqlite3.connect('./chatbot.db')
res = conn.execute('SELECT content FROM chatmessage WHERE content LIKE \'%No relevant documents found%\'').fetchall()
print('FOUND:', len(res))
