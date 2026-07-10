import unittest
from unittest.mock import Mock, patch

from bson import ObjectId

from app.services import ranker_service


class DummyCursor:
    def __init__(self, docs):
        self._docs = docs

    async def to_list(self, length=None):
        return list(self._docs)


class RankerServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_candidate_profile_accepts_string_candidate_ids(self):
        candidate_id = "507f1f77bcf86cd799439011"
        candidate_col = Mock()
        async def _find_one(*args, **kwargs):
            return {
                "_id": ObjectId(candidate_id),
                "fullName": "Alice Nguyen",
                "email": "alice@example.com",
            }

        candidate_col.find_one = Mock(side_effect=_find_one)

        db = Mock()
        db.__getitem__ = Mock(return_value=candidate_col)

        with patch.object(ranker_service, "get_db", return_value=db):
            profile = await ranker_service._get_candidate_profile(candidate_id)

        self.assertEqual(profile["fullName"], "Alice Nguyen")
        self.assertEqual(profile["email"], "alice@example.com")
        self.assertEqual(candidate_col.find_one.call_args[0][0]["_id"], ObjectId(candidate_id))

    async def test_load_target_parsed_resumes_filters_by_employer_jobs(self):
        employer_id = ObjectId("507f1f77bcf86cd799439011")
        job_id = ObjectId("607f1f77bcf86cd799439012")
        resume_id = ObjectId("707f1f77bcf86cd799439013")

        jobs_col = Mock()
        jobs_col.find.return_value = DummyCursor([{"_id": job_id, "employer_id": employer_id}])

        applications_col = Mock()
        applications_col.find.return_value = DummyCursor(
            [{"resumeId": resume_id, "jobId": job_id, "candidateId": ObjectId("807f1f77bcf86cd799439014")}]
        )

        parsed_col = Mock()
        parsed_col.find.return_value = DummyCursor(
            [{"resumeId": resume_id, "shortSummary": "Python backend engineer", "candidateId": ObjectId("807f1f77bcf86cd799439014")}]
        )

        with patch.object(ranker_service, "get_jobs_col", return_value=jobs_col), patch.object(
            ranker_service, "get_applications_col", return_value=applications_col
        ), patch.object(ranker_service, "get_parsed_resumes_col", return_value=parsed_col):
            results = await ranker_service._load_target_parsed_resumes(employer_id=employer_id)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["resumeId"], resume_id)
        self.assertEqual(results[0]["shortSummary"], "Python backend engineer")


if __name__ == "__main__":
    unittest.main()
