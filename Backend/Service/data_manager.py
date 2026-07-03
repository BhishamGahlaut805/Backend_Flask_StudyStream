"""Data Manager - MongoDB-based storage for student data."""

import logging
import pickle
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from bson import ObjectId, Binary

from db import get_db
from config import Config

logger = logging.getLogger(__name__)


class StudentDataManager:
    """Manages student data in MongoDB."""

    def __init__(self, student_id: str):
        self.student_id = student_id
        self.db = get_db()
        self._collections = {
            'practice_features': self.db.get_collection('practice_features'),
            'global_features': self.db.get_collection('global_features'),
            'exam_features': self.db.get_collection('exam_features'),
            'concept_features': self.db.get_collection('concept_features'),
            'interactions': self.db.get_collection('interactions'),
            'predictions': self.db.get_collection('predictions'),
            'schedules': self.db.get_collection('schedules'),
            'performance_metrics': self.db.get_collection('performance_metrics'),
            'sessions': self.db.get_collection('sessions'),
            'model_metadata': self.db.get_collection('model_metadata'),
            'models': self.db.get_collection('models'),
            'micro_sequences': self.db.get_collection('micro_sequences'),
            'meso_sequences': self.db.get_collection('meso_sequences'),
            'macro_sequences': self.db.get_collection('macro_sequences'),
            'daily_aggregates': self.db.get_collection('daily_aggregates'),
            'exam_records': self.db.get_collection('exam_records')
        }
        logger.info(f"Initialized DataManager for student {student_id}")

    # ==================== ADD MISSING SAVE METHODS ====================

    def save_student(self, student_data: Dict) -> bool:
        """Save student details to MongoDB."""
        try:
            if '_id' in student_data:
                del student_data['_id']
            student_data['student_id'] = self.student_id
            self._collections['students'].update_one(
                {'student_id': self.student_id},
                {'$set': student_data},
                upsert=True
            )
            logger.info(f"✅ Saved student record for {self.student_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving student: {e}")
            return False

    def save_performance_metrics(self, metrics: Dict) -> bool:
        """Save performance metrics to MongoDB."""
        try:
            if '_id' in metrics:
                del metrics['_id']
            metrics['student_id'] = self.student_id
            metrics['timestamp'] = datetime.now().isoformat()
            self._collections['performance_metrics'].insert_one(metrics)
            logger.info(f"✅ Saved performance metrics for {self.student_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
            return False

    def save_daily_aggregate(self, aggregate: Dict) -> bool:
        """Save daily aggregate to MongoDB."""
        try:
            if '_id' in aggregate:
                del aggregate['_id']
            aggregate['student_id'] = self.student_id
            aggregate['date'] = aggregate.get('date', datetime.now().strftime('%Y-%m-%d'))
            self._collections['daily_aggregates'].update_one(
                {'student_id': self.student_id, 'date': aggregate['date']},
                {'$set': aggregate},
                upsert=True
            )
            logger.info(f"✅ Saved daily aggregate for {self.student_id} on {aggregate['date']}")
            return True
        except Exception as e:
            logger.error(f"Error saving daily aggregate: {e}")
            return False

    def save_concept_features(self, concept_data: Dict[str, Dict]) -> bool:
        """Save concept features to MongoDB."""
        if not concept_data:
            return False

        try:
            records = []
            for concept, features in concept_data.items():
                record = {
                    'student_id': self.student_id,
                    'concept': concept,
                    **features,
                    'updated_at': datetime.now().isoformat()
                }
                records.append(record)

            # Replace existing concept data
            for record in records:
                self._collections['concept_features'].update_one(
                    {'student_id': self.student_id, 'concept': record['concept']},
                    {'$set': record},
                    upsert=True
                )
            logger.info(f"✅ Saved concept features for {len(records)} concepts")
            return True
        except Exception as e:
            logger.error(f"Error saving concept features: {e}")
            return False

    # ==================== FIX SEQUENCE GENERATION TO SAVE PROPERLY ====================

    # data_manager.py - Fix _generate_micro_sequences_from_features

    def _generate_micro_sequences_from_features(self):
        """Generate micro sequences from practice features."""
        try:
            practice_df = self.load_practice_features()
            if practice_df.empty or len(practice_df) < 20:
                logger.debug(f"Not enough data for micro sequences: {len(practice_df)} rows")
                return

            # Sort by timestamp
            if 'timestamp' in practice_df.columns:
                practice_df = practice_df.sort_values('timestamp')

            seq_length = 20
            # ==================== FIX: Use only numeric columns ====================
            numeric_cols = practice_df.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) < 5:
                logger.warning(f"Not enough numeric columns for micro sequences: {len(numeric_cols)}")
                return

            data = practice_df[numeric_cols].values.astype(np.float32)
            # ==================== END OF FIX ====================

            sequences = []
            for i in range(len(data) - seq_length + 1):
                seq_data = data[i:i + seq_length]
                seq_entry = {
                    'student_id': self.student_id,
                    'sequence_index': i,
                    'sequence_length': seq_length,
                    'features': seq_data.tolist(),
                    'timestamp': datetime.now().isoformat()
                }
                sequences.append(seq_entry)

            if sequences:
                self._collections['micro_sequences'].delete_many({'student_id': self.student_id})
                self._collections['micro_sequences'].insert_many(sequences)
                logger.info(f"✅ Generated {len(sequences)} micro sequences for {self.student_id}")

        except Exception as e:
            logger.error(f"Error generating micro sequences: {e}")

    def _generate_meso_sequences_from_features(self):
        """Generate meso sequences from practice features."""
        try:
            practice_df = self.load_practice_features()
            if practice_df.empty or len(practice_df) < 30:
                logger.debug(f"Not enough data for meso sequences: {len(practice_df)} rows")
                return

            if 'timestamp' in practice_df.columns:
                practice_df = practice_df.sort_values('timestamp')

            if 'concept' not in practice_df.columns:
                practice_df['concept'] = 'general'

            seq_length = 30
            sequences = []

            for concept, group in practice_df.groupby('concept'):
                if len(group) < seq_length:
                    continue

                data = group[Config.PRACTICE_FEATURES].values.astype(np.float32)

                # Create meso-level aggregates (rolling averages)
                for i in range(len(data) - seq_length + 1):
                    seq_data = data[i:i + seq_length]
                    # Compute aggregate features for the sequence
                    agg_features = [
                        float(np.mean(seq_data[:, 0])),  # avg accuracy
                        float(np.mean(seq_data[:, 1])),  # avg normalized response time
                        float(np.mean(seq_data[:, 2])),  # avg time variance
                        float(np.mean(seq_data[:, 3])),  # avg answer changes
                        float(np.mean(seq_data[:, 4])),  # avg stress
                        float(np.mean(seq_data[:, 5])),  # avg confidence
                        float(np.mean(seq_data[:, 6])),  # avg concept mastery
                        float(np.mean(seq_data[:, 7])),  # avg difficulty
                        float(np.mean(seq_data[:, 8])),  # avg streak
                        float(np.mean(seq_data[:, 9])),  # avg fatigue
                        float(np.mean(seq_data[:, 10])), # avg focus loss
                        float(np.mean(seq_data[:, 11]))  # avg difficulty offset
                    ]
                    seq_entry = {
                        'student_id': self.student_id,
                        'concept': str(concept),
                        'sequence_index': i,
                        'sequence_length': seq_length,
                        'features': seq_data.tolist(),
                        'aggregate_features': agg_features,
                        'timestamp': datetime.now().isoformat()
                    }
                    sequences.append(seq_entry)

            if sequences:
                self._collections['meso_sequences'].delete_many({'student_id': self.student_id})
                self._collections['meso_sequences'].insert_many(sequences)
                logger.info(f"✅ Generated {len(sequences)} meso sequences for {self.student_id}")

        except Exception as e:
            logger.error(f"Error generating meso sequences: {e}")

    def _generate_macro_sequences_from_features(self):
        """Generate macro sequences from practice features."""
        try:
            practice_df = self.load_practice_features()
            if practice_df.empty or len(practice_df) < 90:
                logger.debug(f"Not enough data for macro sequences: {len(practice_df)} rows")
                return

            if 'timestamp' in practice_df.columns:
                practice_df = practice_df.sort_values('timestamp')

            seq_length = 90
            data = practice_df[Config.PRACTICE_FEATURES].values.astype(np.float32)

            sequences = []
            for i in range(len(data) - seq_length + 1):
                seq_data = data[i:i + seq_length]
                # Compute macro-level aggregates
                agg_features = {
                    'avg_accuracy': float(np.mean(seq_data[:, 0])),
                    'avg_difficulty': float(np.mean(seq_data[:, 7])),
                    'avg_fatigue': float(np.mean(seq_data[:, 9])),
                    'avg_confidence': float(np.mean(seq_data[:, 5])),
                    'avg_stress': float(np.mean(seq_data[:, 4])),
                    'trend_accuracy': float(seq_data[-1, 0] - seq_data[0, 0]),
                    'trend_difficulty': float(seq_data[-1, 7] - seq_data[0, 7]),
                    'std_accuracy': float(np.std(seq_data[:, 0])),
                    'std_response_time': float(np.std(seq_data[:, 1]))
                }
                seq_entry = {
                    'student_id': self.student_id,
                    'sequence_index': i,
                    'sequence_length': seq_length,
                    'features': seq_data.tolist(),
                    'aggregate_features': agg_features,
                    'timestamp': datetime.now().isoformat()
                }
                sequences.append(seq_entry)

            if sequences:
                self._collections['macro_sequences'].delete_many({'student_id': self.student_id})
                self._collections['macro_sequences'].insert_many(sequences)
                logger.info(f"✅ Generated {len(sequences)} macro sequences for {self.student_id}")

        except Exception as e:
            logger.error(f"Error generating macro sequences: {e}")

    def _student_query(self, extra: Optional[Dict] = None) -> Dict:
        """Base query filter for student data."""
        query = {'student_id': self.student_id}
        if extra:
            query.update(extra)
        return query

    # ==================== PRACTICE FEATURES ====================

    def load_practice_features(self) -> pd.DataFrame:
        """Load practice features from MongoDB."""
        try:
            cursor = self._collections['practice_features'].find(
                self._student_query(),
                {'_id': 0}
            ).sort('timestamp', -1)

            records = list(cursor)
            if records:
                df = pd.DataFrame(records)
                # Ensure all feature columns exist
                from config import Config
                for col in Config.PRACTICE_FEATURES + [Config.PRACTICE_TARGET]:
                    if col not in df.columns:
                        df[col] = 0.5
                logger.debug(f"Loaded {len(df)} practice features for {self.student_id}")
                return df
            logger.debug(f"No practice features found for {self.student_id}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading practice features: {e}")
            return pd.DataFrame()

    def save_practice_features(self, features_df: pd.DataFrame) -> bool:
        """Save practice features to MongoDB."""
        if features_df.empty:
            logger.warning(f"Attempted to save empty practice features for {self.student_id}")
            return False

        try:
            records = features_df.to_dict('records')
            for record in records:
                record['student_id'] = self.student_id
                if 'timestamp' not in record:
                    record['timestamp'] = datetime.now().isoformat()

            if records:
                # ==================== FIX: Use insert_many and verify ====================
                result = self._collections['practice_features'].insert_many(records)
                logger.info(f"✅ Saved {len(records)} practice feature records for {self.student_id}")
                return True
        except Exception as e:
            logger.error(f"Error saving practice features: {e}")
            return False

    def append_practice_attempts_as_features(self, attempts: List[Dict]) -> Dict[str, int]:
        """
        Process attempts and store as features.
        Each attempt is converted to feature vector and saved to MongoDB.
        """
        from Service.feature_engineering import FeatureEngineeringService

        if not attempts:
            logger.warning(f"No attempts to process for {self.student_id}")
            return {'added_rows': 0, 'total_rows': len(self.load_practice_features())}

        logger.info(f"🔄 Processing {len(attempts)} attempts for {self.student_id}")

        try:
            feature_service = FeatureEngineeringService()

            # ==================== FIX: Convert ObjectId to string for serialization ====================
            cleaned_attempts = []
            for attempt in attempts:
                cleaned = {}
                for key, value in attempt.items():
                    if isinstance(value, ObjectId):
                        cleaned[key] = str(value)
                    else:
                        cleaned[key] = value
                cleaned_attempts.append(cleaned)
            # ==================== END OF FIX ====================

            # Compute features from attempts
            features_df = feature_service.compute_practice_features(cleaned_attempts)

            if features_df.empty:
                logger.error(f"❌ Feature computation failed for {self.student_id}")
                logger.error(f"Attempts data: {cleaned_attempts}")
                return {'added_rows': 0, 'total_rows': len(self.load_practice_features())}

            # Add student_id and timestamp
            features_df['student_id'] = self.student_id
            features_df['timestamp'] = datetime.now().isoformat()

            # Save to practice_features
            saved = self.save_practice_features(features_df)

            if saved:
                total_rows = len(self.load_practice_features())
                logger.info(f"✅ Successfully saved {len(features_df)} feature rows for {self.student_id}. Total: {total_rows}")

                # Generate sequences after saving
                self._generate_micro_sequences_from_features()
                self._generate_meso_sequences_from_features()
                self._generate_macro_sequences_from_features()

                return {
                    'added_rows': len(features_df),
                    'total_rows': total_rows
                }
            else:
                logger.error(f"❌ Failed to save features for {self.student_id}")
                return {'added_rows': 0, 'total_rows': len(self.load_practice_features())}

        except Exception as e:
            logger.error(f"❌ Error in append_practice_attempts_as_features: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'added_rows': 0, 'total_rows': len(self.load_practice_features())}

    # ==================== SEQUENCE GENERATION ====================

    def _generate_micro_sequences_from_features(self):
        """Generate micro sequences from practice features."""
        try:
            practice_df = self.load_practice_features()
            if practice_df.empty or len(practice_df) < 20:
                logger.debug(f"Not enough data for micro sequences: {len(practice_df)} rows")
                return

            # Sort by timestamp
            if 'timestamp' in practice_df.columns:
                practice_df = practice_df.sort_values('timestamp')

            seq_length = 20
            feature_cols = Config.PRACTICE_FEATURES

            available_cols = [col for col in feature_cols if col in practice_df.columns]
            if len(available_cols) < 12:
                logger.warning(f"Not enough feature columns for micro sequences")
                return

            data = practice_df[available_cols].values.astype(np.float32)

            sequences = []
            for i in range(len(data) - seq_length + 1):
                seq_data = data[i:i + seq_length]
                seq_entry = {
                    'student_id': self.student_id,
                    'sequence_index': i,
                    'sequence_length': seq_length,
                    'features': seq_data.tolist(),
                    'timestamp': datetime.now().isoformat()
                }
                sequences.append(seq_entry)

            if sequences:
                self._collections['micro_sequences'].delete_many({'student_id': self.student_id})
                self._collections['micro_sequences'].insert_many(sequences)
                logger.info(f"✅ Generated {len(sequences)} micro sequences for {self.student_id}")

        except Exception as e:
            logger.error(f"Error generating micro sequences: {e}")

    def _generate_meso_sequences_from_features(self):
        """Generate meso sequences from practice features."""
        try:
            practice_df = self.load_practice_features()
            if practice_df.empty or len(practice_df) < 30:
                logger.debug(f"Not enough data for meso sequences: {len(practice_df)} rows")
                return

            if 'timestamp' in practice_df.columns:
                practice_df = practice_df.sort_values('timestamp')

            if 'concept' not in practice_df.columns:
                practice_df['concept'] = 'general'

            seq_length = 30
            sequences = []

            for concept, group in practice_df.groupby('concept'):
                if len(group) < seq_length:
                    continue

                data = group[Config.PRACTICE_FEATURES].values.astype(np.float32)

                for i in range(len(data) - seq_length + 1):
                    seq_data = data[i:i + seq_length]
                    seq_entry = {
                        'student_id': self.student_id,
                        'concept': str(concept),
                        'sequence_index': i,
                        'sequence_length': seq_length,
                        'features': seq_data.tolist(),
                        'timestamp': datetime.now().isoformat()
                    }
                    sequences.append(seq_entry)

            if sequences:
                self._collections['meso_sequences'].delete_many({'student_id': self.student_id})
                self._collections['meso_sequences'].insert_many(sequences)
                logger.info(f"✅ Generated {len(sequences)} meso sequences for {self.student_id}")

        except Exception as e:
            logger.error(f"Error generating meso sequences: {e}")

    def _generate_macro_sequences_from_features(self):
        """Generate macro sequences from practice features."""
        try:
            practice_df = self.load_practice_features()
            if practice_df.empty or len(practice_df) < 90:
                logger.debug(f"Not enough data for macro sequences: {len(practice_df)} rows")
                return

            if 'timestamp' in practice_df.columns:
                practice_df = practice_df.sort_values('timestamp')

            seq_length = 90
            data = practice_df[Config.PRACTICE_FEATURES].values.astype(np.float32)

            sequences = []
            for i in range(len(data) - seq_length + 1):
                seq_data = data[i:i + seq_length]
                seq_entry = {
                    'student_id': self.student_id,
                    'sequence_index': i,
                    'sequence_length': seq_length,
                    'features': seq_data.tolist(),
                    'timestamp': datetime.now().isoformat()
                }
                sequences.append(seq_entry)

            if sequences:
                self._collections['macro_sequences'].delete_many({'student_id': self.student_id})
                self._collections['macro_sequences'].insert_many(sequences)
                logger.info(f"✅ Generated {len(sequences)} macro sequences for {self.student_id}")

        except Exception as e:
            logger.error(f"Error generating macro sequences: {e}")

    # ==================== INTERACTIONS ====================

    def load_interactions(self) -> pd.DataFrame:
        """Load interactions from MongoDB."""
        try:
            cursor = self._collections['interactions'].find(
                self._student_query(),
                {'_id': 0}
            ).sort('timestamp', -1)
            records = list(cursor)
            if records:
                logger.debug(f"Loaded {len(records)} interactions for {self.student_id}")
            return pd.DataFrame(records) if records else pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading interactions: {e}")
            return pd.DataFrame()

    def save_interaction(self, interaction: Dict) -> bool:
        """Save a single interaction to MongoDB."""
        try:
            # ==================== FIX: Remove ObjectId if present ====================
            if '_id' in interaction:
                del interaction['_id']
            # ==================== END OF FIX ====================

            interaction['student_id'] = self.student_id
            if 'timestamp' not in interaction:
                interaction['timestamp'] = datetime.now().isoformat()

            self._collections['interactions'].insert_one(interaction)
            logger.debug(f"✅ Saved interaction for {self.student_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving interaction: {e}")
            return False

    def save_interactions(self, interactions: List[Dict]) -> bool:
        """Save multiple interactions to MongoDB."""
        if not interactions:
            return False
        try:
            for interaction in interactions:
                if '_id' in interaction:
                    del interaction['_id']
                interaction['student_id'] = self.student_id
                if 'timestamp' not in interaction:
                    interaction['timestamp'] = datetime.now().isoformat()
            self._collections['interactions'].insert_many(interactions)
            logger.info(f"Saved {len(interactions)} interactions for {self.student_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving interactions: {e}")
            return False

    # ==================== MODEL METADATA ====================

    def load_model_metadata(self, model_name: str) -> List[Dict]:
        """Load model training metadata."""
        try:
            cursor = self._collections['model_metadata'].find(
                {'student_id': self.student_id, 'model_name': model_name},
                {'_id': 0}
            ).sort('timestamp', -1)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error loading model metadata: {e}")
            return []

    def save_model_metadata(self, model_name: str, metadata: Dict) -> bool:
        """Save model training metadata."""
        try:
            record = {
                'student_id': self.student_id,
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                **metadata
            }

            self._collections['model_metadata'].insert_one(record)

            # Keep only last 20 entries
            cursor = self._collections['model_metadata'].find(
                {'student_id': self.student_id, 'model_name': model_name}
            ).sort('timestamp', -1).skip(20)

            for doc in cursor:
                self._collections['model_metadata'].delete_one({'_id': doc['_id']})

            logger.info(f"✅ Saved metadata for {model_name} for student {self.student_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving model metadata: {e}")
            return False

    # ==================== UTILITIES ====================

    def reset_practice_data(self) -> Dict[str, List[str]]:
        """Reset all practice data."""
        cleared = []

        collections = [
            'practice_features',
            'global_features',
            'concept_features',
            'micro_sequences',
            'meso_sequences',
            'macro_sequences',
            'model_metadata'
        ]

        for coll_name in collections:
            try:
                result = self._collections[coll_name].delete_many(
                    self._student_query()
                )
                if result.deleted_count > 0:
                    cleared.append(coll_name)
            except Exception as e:
                logger.error(f"Error clearing {coll_name}: {e}")

        return {'cleared_files': cleared}

    def get_retrain_status(self, model_name: str) -> Dict[str, Any]:
        """Get retrain status for a model."""
        metadata = self.load_model_metadata(model_name)
        if not metadata:
            return {}

        latest = metadata[0] if metadata else {}
        return {
            'last_trained_at': latest.get('timestamp'),
            'last_trained_feature_rows': latest.get('feature_rows_at_training'),
            'training_count': len(metadata)
        }

    # ==================== ADD MISSING METHODS ====================

    def load_global_features(self) -> pd.DataFrame:
        """Load global features from MongoDB."""
        try:
            cursor = self._collections['global_features'].find(
                self._student_query(),
                {'_id': 0}
            )
            records = list(cursor)
            return pd.DataFrame(records) if records else pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading global features: {e}")
            return pd.DataFrame()

    def save_global_features(self, features_df: pd.DataFrame) -> bool:
        """Save global features to MongoDB."""
        if features_df.empty:
            return False
        try:
            records = features_df.to_dict('records')
            for record in records:
                record['student_id'] = self.student_id
            if records:
                self._collections['global_features'].insert_many(records)
                logger.info(f"Saved {len(records)} global feature records")
                return True
        except Exception as e:
            logger.error(f"Error saving global features: {e}")
            return False

    def load_concept_features(self) -> Dict[str, Dict]:
        """Load concept features from MongoDB."""
        try:
            cursor = self._collections['concept_features'].find(
                self._student_query(),
                {'_id': 0}
            )
            records = list(cursor)
            result = {}
            for record in records:
                concept = record.pop('concept', 'unknown')
                record.pop('student_id', None)
                result[concept] = record
            return result
        except Exception as e:
            logger.error(f"Error loading concept features: {e}")
            return {}

    def load_exam_features(self) -> pd.DataFrame:
        """Load exam features from MongoDB."""
        try:
            cursor = self._collections['exam_features'].find(
                self._student_query(),
                {'_id': 0}
            ).sort('timestamp', -1)
            records = list(cursor)
            return pd.DataFrame(records) if records else pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading exam features: {e}")
            return pd.DataFrame()

    def save_exam_features(self, features_df: pd.DataFrame) -> bool:
        """Save exam features to MongoDB."""
        if features_df.empty:
            return False
        try:
            records = features_df.to_dict('records')
            for record in records:
                record['student_id'] = self.student_id
                record['timestamp'] = datetime.now().isoformat()
            if records:
                self._collections['exam_features'].insert_many(records)
                return True
        except Exception as e:
            logger.error(f"Error saving exam features: {e}")
            return False

    def save_exam_records(self, records: List[Dict]) -> bool:
        """Save exam records to MongoDB."""
        if not records:
            return False
        try:
            for record in records:
                if '_id' in record:
                    del record['_id']
                record['student_id'] = self.student_id
                if 'timestamp' not in record:
                    record['timestamp'] = datetime.now().isoformat()
            self._collections['exam_records'].insert_many(records)
            logger.info(f"Saved {len(records)} exam records")
            return True
        except Exception as e:
            logger.error(f"Error saving exam records: {e}")
            return False

    def load_exam_records(self) -> pd.DataFrame:
        """Load exam records from MongoDB."""
        try:
            cursor = self._collections['exam_records'].find(
                self._student_query(),
                {'_id': 0}
            ).sort('timestamp', -1)
            records = list(cursor)
            return pd.DataFrame(records) if records else pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading exam records: {e}")
            return pd.DataFrame()

    def save_model(self, model_name: str, model: Any, metadata: Dict) -> str:
        """Save model to MongoDB using GridFS."""
        try:
            import pickle
            model_data = pickle.dumps(model)
            file_id = self.db.fs.put(
                model_data,
                filename=f"{model_name}_{self.student_id}",
                metadata={
                    'model_name': model_name,
                    'student_id': self.student_id,
                    **metadata,
                    'saved_at': datetime.now().isoformat()
                }
            )
            logger.info(f"Saved model {model_name} for student {self.student_id} with ID: {file_id}")
            return str(file_id)
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            raise

    def load_latest_model(self, model_name: str, student_id: str) -> Any:
        """Load latest model from GridFS."""
        try:
            import pickle
            files = list(self.db.fs.find({
                'metadata.model_name': model_name,
                'metadata.student_id': student_id
            }).sort('uploadDate', -1).limit(1))
            if not files:
                return None
            model_data = files[0].read()
            return pickle.loads(model_data)
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None

    # ==================== SEQUENCE LOAD METHODS ====================

    # data_manager.py - Fix load_micro_sequences to return numeric data only

    def load_micro_sequences(self) -> pd.DataFrame:
        """Load micro sequences from MongoDB."""
        try:
            cursor = self._collections['micro_sequences'].find(
                self._student_query(),
                {'_id': 0}
            ).sort('timestamp', -1)
            records = list(cursor)

            if not records:
                logger.debug(f"No micro sequences found for {self.student_id}")
                return pd.DataFrame()

            # ==================== FIX: Process records to extract numeric features ====================
            processed_records = []
            for record in records:
                processed = {}
                # Keep metadata
                for key in ['student_id', 'sequence_index', 'sequence_length', 'timestamp', 'topic_id', 'concept']:
                    if key in record:
                        processed[key] = record[key]

                # Extract numeric features
                if 'features' in record and isinstance(record['features'], list):
                    try:
                        # Convert features to numeric array
                        features_array = np.array(record['features']).astype(np.float32)
                        # Add each feature as a separate column
                        for i, val in enumerate(features_array.flatten()[:20]):  # Limit to 20 features
                            processed[f'feature_{i}'] = float(val)
                    except Exception as e:
                        logger.debug(f"Could not extract features from record: {e}")

                processed_records.append(processed)

            df = pd.DataFrame(processed_records)
            logger.debug(f"Loaded {len(df)} micro sequences for {self.student_id}")
            return df
            # ==================== END OF FIX ====================

        except Exception as e:
            logger.error(f"Error loading micro sequences: {e}")
            return pd.DataFrame()

    def load_meso_sequences(self) -> pd.DataFrame:
        """Load meso sequences from MongoDB."""
        try:
            cursor = self._collections['meso_sequences'].find(
                self._student_query(),
                {'_id': 0}
            ).sort('timestamp', -1)
            records = list(cursor)
            if records:
                logger.debug(f"Loaded {len(records)} meso sequences for {self.student_id}")
            return pd.DataFrame(records) if records else pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading meso sequences: {e}")
            return pd.DataFrame()

    def load_macro_sequences(self) -> pd.DataFrame:
        """Load macro sequences from MongoDB."""
        try:
            cursor = self._collections['macro_sequences'].find(
                self._student_query(),
                {'_id': 0}
            ).sort('timestamp', -1)
            records = list(cursor)
            if records:
                logger.debug(f"Loaded {len(records)} macro sequences for {self.student_id}")
            return pd.DataFrame(records) if records else pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading macro sequences: {e}")
            return pd.DataFrame()

    def load_predictions(self, model_type: str) -> Any:
        """Load predictions for a model type."""
        try:
            cursor = self._collections['predictions'].find(
                {'student_id': self.student_id, 'model_type': model_type},
                {'_id': 0}
            ).sort('timestamp', -1).limit(1)
            records = list(cursor)
            return records[0].get('predictions', []) if records else []
        except Exception as e:
            logger.error(f"Error loading predictions for {model_type}: {e}")
            return []

    def save_predictions(self, model_type: str, predictions: Any) -> bool:
        """Save predictions for a model type."""
        try:
            record = {
                'student_id': self.student_id,
                'model_type': model_type,
                'predictions': predictions,
                'timestamp': datetime.now().isoformat()
            }
            self._collections['predictions'].update_one(
                {'student_id': self.student_id, 'model_type': model_type},
                {'$set': record},
                upsert=True
            )
            logger.info(f"Saved predictions for {model_type} for student {self.student_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving predictions for {model_type}: {e}")
            return False

    def load_daily_aggregates(self) -> pd.DataFrame:
        """Load daily aggregates from MongoDB."""
        try:
            cursor = self._collections['daily_aggregates'].find(
                self._student_query(),
                {'_id': 0}
            ).sort('date', -1)
            records = list(cursor)
            return pd.DataFrame(records) if records else pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading daily aggregates: {e}")
            return pd.DataFrame()

    def save_session(self, session_data: Dict) -> bool:
        """Save session data to MongoDB."""
        try:
            if '_id' in session_data:
                del session_data['_id']
            session_data['student_id'] = self.student_id
            self._collections['sessions'].update_one(
                {'student_id': self.student_id, 'session_id': session_data.get('session_id')},
                {'$set': session_data},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            return False

    def load_sessions(self, limit: int = 100) -> List[Dict]:
        """Load recent sessions."""
        try:
            cursor = self._collections['sessions'].find(
                self._student_query(),
                {'_id': 0}
            ).sort('started_at', -1).limit(limit)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
            return []


    # data_manager.py - Add these methods to the StudentDataManager class

    # ==================== TRAINING DATA PREPARATION METHODS ====================
    # Add these methods after the existing methods

    def prepare_practice_training_data(self, min_samples: int) -> Optional[Dict]:
        """
        Prepare training data for practice difficulty model.

        Args:
            min_samples: Minimum number of samples required

        Returns:
            Dictionary with training data or None if insufficient
        """
        df = self.load_practice_features()

        if df.empty:
            logger.warning(f"No practice features found for {self.student_id}")
            return None

        logger.info(f"Preparing practice training data for {self.student_id}. Available samples: {len(df)}")

        if len(df) < min_samples:
            logger.warning(f"Insufficient practice data: {len(df)} < {min_samples}")
            return None

        from config import Config

        feature_cols = Config.PRACTICE_FEATURES
        target_col = Config.PRACTICE_TARGET

        # Verify all columns exist
        available_cols = [col for col in feature_cols if col in df.columns]
        if len(available_cols) < 10:
            logger.warning(f"Insufficient feature columns. Found: {available_cols}")
            return None

        # Extract data
        data_values = df[available_cols].values.astype(np.float32)
        target_values = df[target_col].values.astype(np.float32)

        seq_length = Config.SEQUENCE_LENGTH_PRACTICE

        # Create sequences
        X, y = [], []
        for i in range(len(data_values) - seq_length):
            X.append(data_values[i:i + seq_length])
            y.append(target_values[i + seq_length])

        if len(X) < 5:
            logger.warning(f"Insufficient sequences: {len(X)} < 5")
            return None

        X, y = np.array(X), np.array(y)
        logger.info(f"Created {len(X)} training sequences")

        # Split into train/val/test
        n = len(X)
        indices = np.random.permutation(n)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end] if val_end > train_end else []
        test_idx = indices[val_end:] if n > val_end else []

        result = {
            'X_train': X[train_idx],
            'y_train': y[train_idx],
            'feature_names': available_cols,
            'total_samples': len(X)
        }

        if len(val_idx) > 0:
            result['X_val'] = X[val_idx]
            result['y_val'] = y[val_idx]

        if len(test_idx) > 0:
            result['X_test'] = X[test_idx]
            result['y_test'] = y[test_idx]

        return result

    def prepare_exam_training_data(self, min_samples: int) -> Optional[Dict]:
        """
        Prepare training data for exam difficulty model.

        Args:
            min_samples: Minimum number of samples required

        Returns:
            Dictionary with training data or None if insufficient
        """
        df = self.load_exam_features()

        if df.empty:
            logger.warning(f"No exam features found for {self.student_id}")
            return None

        if len(df) < min_samples:
            logger.warning(f"Insufficient exam data: {len(df)} < {min_samples}")
            return None

        from config import Config

        feature_cols = [
            'overall_accuracy_avg', 'avg_difficulty_handled',
            'readiness_score', 'consistency_index',
            'exam_performance_trend', 'concept_coverage_ratio',
            'time_efficiency_score', 'stamina_index'
        ]

        available_cols = [col for col in feature_cols if col in df.columns]
        if len(available_cols) < 4:
            logger.warning(f"Insufficient exam feature columns. Found: {available_cols}")
            return None

        seq_length = Config.SEQUENCE_LENGTH_EXAM
        data_values = df[available_cols].values.astype(np.float32)
        target_values = df[available_cols].values.astype(np.float32)

        X, y = [], []
        for i in range(len(data_values) - seq_length):
            X.append(data_values[i:i + seq_length])
            y.append(target_values[i + seq_length])

        if len(X) < 3:
            logger.warning(f"Insufficient exam sequences: {len(X)} < 3")
            return None

        X, y = np.array(X), np.array(y)
        n = len(X)
        split = int(n * 0.8)

        return {
            'X_train': X[:split],
            'y_train': y[:split],
            'X_val': X[split:] if split < n else None,
            'y_val': y[split:] if split < n else None,
            'feature_names': available_cols,
            'total_samples': len(X)
        }

    def prepare_global_training_data(self, min_samples: int) -> Optional[Dict]:
        """
        Prepare training data for global readiness model.

        Args:
            min_samples: Minimum number of samples required

        Returns:
            Dictionary with training data or None if insufficient
        """
        df = self.load_global_features()

        if df.empty:
            logger.warning(f"No global features found for {self.student_id}")
            return None

        if len(df) < min_samples:
            logger.warning(f"Insufficient global data: {len(df)} < {min_samples}")
            return None

        from config import Config

        feature_cols = Config.GLOBAL_FEATURES
        target_col = Config.GLOBAL_TARGET

        available_cols = [col for col in feature_cols if col in df.columns]
        if len(available_cols) < 8:
            logger.warning(f"Insufficient global feature columns. Found: {available_cols}")
            return None

        seq_length = Config.SEQUENCE_LENGTH_GLOBAL
        data_values = df[available_cols].values.astype(np.float32)
        target_values = df[target_col].values.astype(np.float32)

        X, y = [], []
        for i in range(len(data_values) - seq_length):
            X.append(data_values[i:i + seq_length])
            y.append(target_values[i + seq_length])

        if len(X) < 3:
            logger.warning(f"Insufficient global sequences: {len(X)} < 3")
            return None

        X, y = np.array(X), np.array(y)
        n = len(X)
        split = int(n * 0.8)

        return {
            'X_train': X[:split],
            'y_train': y[:split],
            'X_val': X[split:] if split < n else None,
            'y_val': y[split:] if split < n else None,
            'feature_names': available_cols,
            'total_samples': len(X)
        }

    def prepare_learning_velocity_training_data(self, concept: str, min_samples: int) -> Optional[Dict]:
        """
        Prepare training data for learning velocity model for a specific concept.

        Args:
            concept: The concept name
            min_samples: Minimum number of samples required

        Returns:
            Dictionary with training data or None if insufficient
        """
        concept_features = self.load_concept_features()

        if concept not in concept_features:
            logger.warning(f"No concept features found for {concept}")
            return None

        feat = concept_features[concept]
        mastery_history = feat.get('concept_mastery_history', [])

        if len(mastery_history) < min_samples:
            logger.warning(f"Insufficient history for {concept}: {len(mastery_history)} < {min_samples}")
            return None

        from config import Config

        seq_length = Config.SEQUENCE_LENGTH_DAILY
        X, y = [], []

        for i in range(len(mastery_history) - seq_length):
            seq = mastery_history[i:i + seq_length]
            feature_seq = []
            for mastery in seq:
                feature_seq.append([
                    float(mastery),
                    float(feat.get('practice_frequency', 1.0)),
                    float(feat.get('revision_gap', 0.5)),
                    float(feat.get('avg_difficulty', 0.6)),
                    float(feat.get('success_rate', 0.7)),
                    float(feat.get('retention', 0.8)),
                    float(feat.get('time_spent', 30)),
                    float(feat.get('improvement_rate', 0.1)),
                    float(feat.get('confidence_growth', 0.6))
                ])
            X.append(feature_seq)
            y.append(mastery_history[i + seq_length])

        if len(X) < 3:
            logger.warning(f"Insufficient learning velocity sequences: {len(X)} < 3")
            return None

        X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        n = len(X)
        split = int(n * 0.8)

        return {
            'X_train': X[:split],
            'y_train': y[:split],
            'X_val': X[split:] if split < n else None,
            'y_val': y[split:] if split < n else None,
            'concept': concept,
            'total_samples': len(X)
        }

    def prepare_burnout_training_data(self, min_samples: int) -> Optional[Dict]:
        """
        Prepare training data for burnout risk model.

        Args:
            min_samples: Minimum number of samples required

        Returns:
            Dictionary with training data or None if insufficient
        """
        df = self.load_practice_features()

        if df.empty or len(df) < min_samples:
            logger.warning(f"Insufficient practice data for burnout: {len(df) if not df.empty else 0} < {min_samples}")
            return None

        from config import Config

        # Create session features
        sessions = []
        if 'session_id' in df.columns:
            unique_sessions = df['session_id'].unique()

            for session_id in unique_sessions:
                session_data = df[df['session_id'] == session_id].sort_values('timestamp')

                if len(session_data) >= 5:
                    try:
                        features = [
                            float(session_data['accuracy'].mean()),
                            float(session_data['accuracy'].diff().mean()) if len(session_data) > 1 else 0,
                            float(session_data['stress_score'].diff().mean()) if 'stress_score' in session_data.columns and len(session_data) > 1 else 0,
                            float(session_data['normalized_response_time'].diff().mean()) if 'normalized_response_time' in session_data.columns and len(session_data) > 1 else 0,
                            float(session_data['fatigue_indicator'].iloc[-1] - session_data['fatigue_indicator'].iloc[0]) if 'fatigue_indicator' in session_data.columns and len(session_data) > 1 else 0,
                            float(session_data['time_spent'].sum() / 60) if 'time_spent' in session_data.columns else 10,
                            1.0,
                            float(session_data[session_data['current_question_difficulty'] > 0.7]['accuracy'].mean()) if 'current_question_difficulty' in session_data.columns and len(session_data[session_data['current_question_difficulty'] > 0.7]) > 0 else 0.5,
                            float(1 - session_data['accuracy'].std()) if len(session_data) > 1 else 0.5,
                            float(session_data['confidence_index'].diff().mean()) if 'confidence_index' in session_data.columns and len(session_data) > 1 else 0,
                            float((session_data['time_spent'] < 5).mean()) if 'time_spent' in session_data.columns else 0,
                            float(session_data['accuracy'].iloc[-len(session_data)//2:].mean() - session_data['accuracy'].iloc[:len(session_data)//2].mean()) if len(session_data) >= 4 else 0
                        ]

                        # Synthetic label (simplified)
                        fatigue_high = session_data['fatigue_indicator'].iloc[-1] > 0.7 if 'fatigue_indicator' in session_data.columns else False
                        accuracy_dropping = (session_data['accuracy'].iloc[-3:].mean() < session_data['accuracy'].iloc[:3].mean()) if len(session_data) >= 6 else False
                        label = 1 if (fatigue_high and accuracy_dropping) else 0

                        sessions.append({
                            'features': features,
                            'label': label
                        })
                    except Exception as e:
                        logger.debug(f"Error processing session {session_id}: {e}")
                        continue

        if len(sessions) < 10:
            logger.warning(f"Insufficient sessions for burnout training: {len(sessions)} < 10")
            return None

        X = np.array([s['features'] for s in sessions], dtype=np.float32)
        y = np.array([s['label'] for s in sessions], dtype=np.float32)

        # Create sequences
        seq_length = Config.SEQUENCE_LENGTH_SESSION
        X_seq, y_seq = [], []

        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i + seq_length])
            y_seq.append(y[i + seq_length])

        if len(X_seq) < 3:
            logger.warning(f"Insufficient burnout sequences: {len(X_seq)} < 3")
            return None

        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        n = len(X_seq)
        split = int(n * 0.8)

        return {
            'X_train': X_seq[:split],
            'y_train': y_seq[:split],
            'X_val': X_seq[split:] if split < n else None,
            'y_val': y_seq[split:] if split < n else None,
            'total_samples': len(X_seq)
        }

    def prepare_adaptive_scheduling_training_data(self, min_samples: int) -> Optional[Dict]:
        """
        Prepare training data for adaptive scheduling model.

        Args:
            min_samples: Minimum number of samples required

        Returns:
            Dictionary with training data or None if insufficient
        """
        concept_features = self.load_concept_features()

        if not concept_features or len(concept_features) < min_samples:
            logger.warning(f"Insufficient concept data: {len(concept_features)} < {min_samples}")
            return None

        X, y = [], []

        for concept, feat in concept_features.items():
            feature_vector = [
                float(feat.get('accuracy', 0.5)),
                float(feat.get('exam_weight', 0.5)),
                float(feat.get('avg_difficulty', 0.5)) * 2,
                float(feat.get('learning_velocity', 0)),
                float(feat.get('stability', 0.5)),
                float(feat.get('readiness', 0.5))
            ]

            # Pad to 13 features
            while len(feature_vector) < 13:
                feature_vector.append(0.5)

            # Create synthetic priority score
            priority = (
                (1 - feat.get('accuracy', 0.5)) * 0.4 +
                feat.get('exam_weight', 0.5) * 0.3 +
                (feat.get('days_since_last_practice', 0) / 30) * 0.3
            )
            priority = min(1.0, max(0.0, priority))

            X.append(feature_vector)
            y.append(priority)

        if len(X) < 3:
            return None

        X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        n = len(X)
        split = int(n * 0.8)

        return {
            'X_train': X[:split],
            'y_train': y[:split],
            'X_val': X[split:] if split < n else None,
            'y_val': y[split:] if split < n else None,
            'total_samples': len(X)
        }
