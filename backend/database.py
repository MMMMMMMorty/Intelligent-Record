"""
Database module for Intelligent Document System
Uses SQLite for simplicity and portability
"""

import aiosqlite
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import os

DATABASE_PATH = os.getenv("DATABASE_PATH", "/app/data/intelligent_record.db")


class Database:
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path

    async def init(self):
        """Initialize database tables"""
        async with aiosqlite.connect(self.db_path) as db:
            # Users table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Transcriptions table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    title TEXT,
                    audio_path TEXT,
                    text TEXT,
                    language TEXT,
                    duration_seconds REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)

            # Document records table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS document_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transcription_id INTEGER,
                    title TEXT NOT NULL,
                    status TEXT DEFAULT 'draft',
                    extracted_info TEXT,
                    record_content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (transcription_id) REFERENCES transcriptions (id)
                )
            """)

            await db.commit()

    @asynccontextmanager
    async def get_db(self):
        """Get database connection context manager"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            yield db

    async def create_user(self, username: str, hashed_password: str) -> int:
        """Create a new user, return user_id"""
        async with self.get_db() as db:
            cursor = await db.execute(
                "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
                (username, hashed_password)
            )
            await db.commit()
            return cursor.lastrowid

    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        async with self.get_db() as db:
            cursor = await db.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,)
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def create_transcription(
        self,
        user_id: Optional[int],
        title: Optional[str],
        audio_path: Optional[str],
        text: str,
        language: Optional[str] = None,
        duration_seconds: Optional[float] = None
    ) -> int:
        """Create a new transcription record"""
        async with self.get_db() as db:
            cursor = await db.execute(
                """INSERT INTO transcriptions
                    (user_id, title, audio_path, text, language, duration_seconds)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, title, audio_path, text, language, duration_seconds)
            )
            await db.commit()
            return cursor.lastrowid

    async def get_transcriptions(self, user_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get transcription records"""
        async with self.get_db() as db:
            if user_id:
                cursor = await db.execute(
                    """SELECT * FROM transcriptions
                        WHERE user_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?""",
                    (user_id, limit)
                )
            else:
                cursor = await db.execute(
                    """SELECT * FROM transcriptions
                        ORDER BY created_at DESC
                        LIMIT ?""",
                    (limit,)
                )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_transcription(self, transcription_id: int) -> Optional[Dict[str, Any]]:
        """Get a single transcription by ID"""
        async with self.get_db() as db:
            cursor = await db.execute(
                "SELECT * FROM transcriptions WHERE id = ?",
                (transcription_id,)
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def delete_transcription(self, transcription_id: int, user_id: Optional[int] = None) -> bool:
        """Delete a transcription record"""
        async with self.get_db() as db:
            if user_id:
                cursor = await db.execute(
                    "DELETE FROM transcriptions WHERE id = ? AND user_id = ?",
                    (transcription_id, user_id)
                )
            else:
                cursor = await db.execute(
                    "DELETE FROM transcriptions WHERE id = ?",
                    (transcription_id,)
                )
            await db.commit()
            return cursor.rowcount > 0

    async def create_document_record(
        self,
        transcription_id: int,
        title: str,
        status: str = "draft"
    ) -> int:
        """创建文档记录"""
        async with self.get_db() as db:
            cursor = await db.execute(
                """INSERT INTO document_records
                    (transcription_id, title, status, created_at, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
                (transcription_id, title, status)
            )
            await db.commit()
            return cursor.lastrowid

    async def get_document_record(self, record_id: int) -> Optional[Dict[str, Any]]:
        """获取文档详情"""
        async with self.get_db() as db:
            cursor = await db.execute(
                """SELECT dr.*, t.text as transcription_text
                   FROM document_records dr
                   LEFT JOIN transcriptions t ON dr.transcription_id = t.id
                   WHERE dr.id = ?""",
                (record_id,)
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def update_document_record(
        self,
        record_id: int,
        extracted_info: Optional[str] = None,
        record_content: Optional[str] = None,
        status: Optional[str] = None
    ) -> bool:
        """更新文档内容"""
        async with self.get_db() as db:
            updates = []
            params = []
            if extracted_info is not None:
                updates.append("extracted_info = ?")
                params.append(extracted_info)
            if record_content is not None:
                updates.append("record_content = ?")
                params.append(record_content)
            if status is not None:
                updates.append("status = ?")
                params.append(status)
            
            if not updates:
                return False
                
            params.append(record_id)
            query = f"""UPDATE document_records
                       SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP
                       WHERE id = ?"""
            
            cursor = await db.execute(query, params)
            await db.commit()
            return cursor.rowcount > 0

    async def list_document_records(
        self,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """获取文档列表"""
        async with self.get_db() as db:
            if status:
                cursor = await db.execute(
                    """SELECT dr.*, t.text as transcription_text
                       FROM document_records dr
                       LEFT JOIN transcriptions t ON dr.transcription_id = t.id
                       WHERE dr.status = ?
                       ORDER BY dr.created_at DESC
                       LIMIT ?""",
                    (status, limit)
                )
            else:
                cursor = await db.execute(
                    """SELECT dr.*, t.text as transcription_text
                       FROM document_records dr
                       LEFT JOIN transcriptions t ON dr.transcription_id = t.id
                       ORDER BY dr.created_at DESC
                       LIMIT ?""",
                    (limit,)
                )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def delete_document_record(self, record_id: int) -> bool:
        """删除文档"""
        async with self.get_db() as db:
            cursor = await db.execute(
                "DELETE FROM document_records WHERE id = ?",
                (record_id,)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def _init_document_record_tables(self):
        """初始化文档相关表"""
        async with self.get_db() as db:
            # 文档表
            await db.execute("""
                CREATE TABLE IF NOT EXISTS document_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transcription_id INTEGER,
                    title TEXT NOT NULL,
                    status TEXT DEFAULT 'draft',
                    extracted_info TEXT,
                    record_content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (transcription_id) REFERENCES transcriptions (id)
                )
            """)
            await db.commit()


# Global database instance
db = Database()
