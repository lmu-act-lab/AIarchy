from student import Student
import unittest
import numpy as np
import copy
import warnings

warnings.filterwarnings("ignore")


def is_array_greater(arr1, arr2):
    return np.any(np.greater(arr1, arr2))


class TestTraining(unittest.TestCase):

    def test_sleep_increase(self):
        pass

    def test_CPT_increase(self):
        low_ses_student = Student({"SES": 0}, threshold=0.00001)
        low_ses_student.train(1)
        self.assertTrue(
            is_array_greater(
                low_ses_student.get_cpt_vals("ECs", 1),
                low_ses_student.get_original_cpt_vals("ECs", 1),
            )
            or is_array_greater(
                low_ses_student.get_cpt_vals("Sleep", 2),
                low_ses_student.get_original_cpt_vals("Sleep", 2),
            )
            or is_array_greater(
                low_ses_student.get_cpt_vals("Time studying", 2),
                low_ses_student.get_original_cpt_vals("Time studying", 2),
            )
            or is_array_greater(
                low_ses_student.get_cpt_vals("Exercise", 1),
                low_ses_student.get_original_cpt_vals("Exercise", 1),
            )
        )

    def test_weight_change(self):
        low_ses_student = Student({"SES": 0}, threshold=1, downweigh_factor=0.0001)
        low_ses_student_before = copy.deepcopy(low_ses_student)
        low_ses_student.train(15)
        self.assertTrue(low_ses_student.weights != low_ses_student_before.weights)


if __name__ == "__main__":
    unittest.main()
