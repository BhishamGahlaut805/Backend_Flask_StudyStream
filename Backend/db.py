"""Database connection and utility functions."""

import logging
import pickle
from typing import Optional, Dict, Any, List
from datetime import datetime

import gridfs
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from bson import ObjectId, Binary

from config import get_config

logger = logging.getLogger(__name__)


class MongoDB:
    """MongoDB connection manager with GridFS support."""

    _instance: Optional['MongoDB'] = None
    _client: Optional[MongoClient] = None
    _db: Optional[Database] = None
    _fs: Optional[gridfs.GridFS] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self._initialize()

    def _initialize(self):
        """Initialize the MongoDB connection."""
        config = get_config()
        try:
            self._client = MongoClient(config.MONGODB_URI)
            self._db = self._client[config.MONGODB_DB_NAME]
            self._fs = gridfs.GridFS(self._db)

            # Create indexes
            self._create_indexes()

            logger.info(f"Connected to MongoDB database: {config.MONGODB_DB_NAME}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def _create_indexes(self):
        """Create necessary indexes for collections."""
        # Students collection
        self._db.students.create_index('student_id', unique=True)
        self._db.students.create_index('email')

        # Practice features collection
        self._db.practice_features.create_index([
            ('student_id', 1),
            ('timestamp', -1)
        ])

        # Global features collection
        self._db.global_features.create_index([
            ('student_id', 1),
            ('session_id', 1)
        ])

        # Exam features collection
        self._db.exam_features.create_index([
            ('student_id', 1),
            ('timestamp', -1)
        ])

        # Concept features collection
        self._db.concept_features.create_index([
            ('student_id', 1),
            ('concept', 1)
        ])

        # Model metadata collection
        self._db.model_metadata.create_index([
            ('student_id', 1),
            ('model_name', 1),
            ('timestamp', -1)
        ])

        # Models collection (for direct model storage)
        self._db.models.create_index([
            ('student_id', 1),
            ('model_name', 1),
            ('timestamp', -1)
        ])

        # Interactions collection
        self._db.interactions.create_index([
            ('student_id', 1),
            ('timestamp', -1)
        ])
        self._db.interactions.create_index([
            ('student_id', 1),
            ('session_id', 1)
        ])

        # Predictions collection
        self._db.predictions.create_index([
            ('student_id', 1),
            ('model_type', 1)
        ])

        # Schedules collection
        self._db.schedules.create_index([
            ('student_id', 1),
            ('date', 1)
        ])

        # Performance metrics collection
        self._db.performance_metrics.create_index([
            ('student_id', 1),
            ('timestamp', -1)
        ])

        # Sessions collection
        self._db.sessions.create_index([
            ('student_id', 1),
            ('session_id', 1)
        ])
        self._db.sessions.create_index([
            ('student_id', 1),
            ('started_at', -1)
        ])

        # Exam records collection
        self._db.exam_records.create_index([
            ('student_id', 1),
            ('timestamp', -1)
        ])

        # Daily aggregates collection
        self._db.daily_aggregates.create_index([
            ('student_id', 1),
            ('date', 1)
        ])

        # Micro/Meso/Macro sequences collections
        self._db.micro_sequences.create_index([
            ('student_id', 1),
            ('timestamp', -1)
        ])
        self._db.meso_sequences.create_index([
            ('student_id', 1),
            ('timestamp', -1)
        ])
        self._db.macro_sequences.create_index([
            ('student_id', 1),
            ('timestamp', -1)
        ])

        logger.info("Database indexes created")

    @property
    def client(self) -> MongoClient:
        """Get MongoDB client."""
        return self._client

    @property
    def db(self) -> Database:
        """Get MongoDB database."""
        return self._db

    @property
    def fs(self) -> gridfs.GridFS:
        """Get GridFS instance for file storage."""
        return self._fs

    def get_collection(self, name: str) -> Collection:
        """Get a collection by name."""
        return self._db[name]

    # ==================== MODEL STORAGE METHODS ====================

    def save_model(self, model_name: str, model_data: Any, metadata: Dict[str, Any]) -> str:
        """
        Save a model to GridFS.

        Args:
            model_name: Name of the model
            model_data: The model object to save
            metadata: Additional metadata to store with the model

        Returns:
            str: The GridFS file ID
        """
        try:
            binary_data = Binary(pickle.dumps(model_data))

            file_id = self._fs.put(
                binary_data,
                filename=f"{model_name}_{metadata.get('student_id', 'unknown')}",
                metadata={
                    'model_name': model_name,
                    **metadata,
                    'saved_at': datetime.now().isoformat()
                }
            )
            logger.info(f"Saved model {model_name} with ID: {file_id}")
            return str(file_id)
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
            raise

    def load_model(self, file_id: str):
        """
        Load a model from GridFS by file ID.

        Args:
            file_id: The GridFS file ID

        Returns:
            The deserialized model object
        """
        try:
            file_data = self._fs.get(ObjectId(file_id))
            model = pickle.loads(file_data.read())
            logger.info(f"Loaded model from ID: {file_id}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {file_id}: {e}")
            raise

    def get_latest_model(self, model_name: str, student_id: Optional[str] = None):
        """
        Get the latest version of a model for a student.

        Args:
            model_name: Name of the model
            student_id: Optional student ID to filter by

        Returns:
            The deserialized model object or None if not found
        """
        try:
            query = {'metadata.model_name': model_name}
            if student_id:
                query['metadata.student_id'] = student_id

            files = list(self._fs.find(query).sort('uploadDate', -1).limit(1))
            if not files:
                logger.info(f"No model found for {model_name} for student {student_id}")
                return None

            return self.load_model(files[0]._id)
        except Exception as e:
            logger.error(f"Failed to get latest model {model_name}: {e}")
            return None

    def delete_model(self, model_name: str, student_id: str) -> bool:
        """
        Delete all models for a student.

        Args:
            model_name: Name of the model
            student_id: Student ID

        Returns:
            bool: True if successful
        """
        try:
            files = list(self._fs.find({
                'metadata.model_name': model_name,
                'metadata.student_id': student_id
            }))

            for file in files:
                self._fs.delete(file._id)

            logger.info(f"Deleted {len(files)} models for {model_name} for student {student_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting models: {e}")
            return False

    def list_models(self, student_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all models for a student.

        Args:
            student_id: Optional student ID to filter by

        Returns:
            List of model metadata
        """
        try:
            query = {}
            if student_id:
                query['metadata.student_id'] = student_id

            files = list(self._fs.find(query).sort('uploadDate', -1))

            models = []
            for file in files:
                models.append({
                    'id': str(file._id),
                    'model_name': file.metadata.get('model_name'),
                    'student_id': file.metadata.get('student_id'),
                    'saved_at': file.metadata.get('saved_at'),
                    'upload_date': file.uploadDate.isoformat() if file.uploadDate else None,
                    'file_size': file.length
                })

            return models
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    # ==================== DATA COLLECTION HELPERS ====================

    def get_or_create_collection(self, name: str) -> Collection:
        """Get a collection, creating it if it doesn't exist."""
        return self._db[name]

    def insert_document(self, collection_name: str, document: Dict[str, Any]) -> str:
        """Insert a document into a collection."""
        try:
            result = self._db[collection_name].insert_one(document)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error inserting document into {collection_name}: {e}")
            raise

    def find_documents(self, collection_name: str, query: Dict[str, Any],
                       limit: int = 0, sort_by: Optional[tuple] = None) -> List[Dict]:
        """Find documents in a collection."""
        try:
            cursor = self._db[collection_name].find(query)
            if sort_by:
                cursor = cursor.sort(*sort_by)
            if limit > 0:
                cursor = cursor.limit(limit)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error finding documents in {collection_name}: {e}")
            return []

    def update_document(self, collection_name: str, query: Dict[str, Any],
                        update: Dict[str, Any], upsert: bool = False) -> bool:
        """Update a document in a collection."""
        try:
            result = self._db[collection_name].update_one(query, {'$set': update}, upsert=upsert)
            return result.modified_count > 0 or result.upserted_id is not None
        except Exception as e:
            logger.error(f"Error updating document in {collection_name}: {e}")
            return False

    def delete_documents(self, collection_name: str, query: Dict[str, Any]) -> int:
        """Delete documents from a collection."""
        try:
            result = self._db[collection_name].delete_many(query)
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error deleting documents from {collection_name}: {e}")
            return 0

    # ==================== CONNECTION MANAGEMENT ====================

    def close(self):
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self._fs = None
            logger.info("MongoDB connection closed")

    def is_connected(self) -> bool:
        """Check if the database connection is active."""
        try:
            if self._client:
                self._client.admin.command('ping')
                return True
            return False
        except Exception:
            return False

    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the current database."""
        try:
            info = {
                'database_name': self._db.name if self._db else None,
                'is_connected': self.is_connected(),
                'collections': list(self._db.list_collection_names()) if self._db else []
            }
            return info
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {'is_connected': False, 'error': str(e)}


# ==================== SINGLETON INSTANCE ====================

_db_instance: Optional[MongoDB] = None


def get_db() -> MongoDB:
    """
    Get the MongoDB instance (singleton).

    Returns:
        MongoDB: The database connection instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = MongoDB()
    return _db_instance


def init_db() -> MongoDB:
    """
    Initialize the database connection.

    Returns:
        MongoDB: The database connection instance
    """
    global _db_instance
    if _db_instance is not None:
        _db_instance.close()
    _db_instance = MongoDB()
    return _db_instance


def close_db():
    """Close the database connection."""
    global _db_instance
    if _db_instance:
        _db_instance.close()
        _db_instance = None
        logger.info("Database connection closed")



