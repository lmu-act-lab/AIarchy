from student import Student
import unittest
import numpy as np


low_ses_student = Student(
    {"SES": 0}, weights={"health": 0.33, "social": 0.33, "grades": 0.33}
)
high_ses_student = Student(
    {"SES": 2}, weights={"health": 0.33, "social": 0.33, "grades": 0.33}
)
high_ses_student.train(100)
low_ses_student.train(100)


def is_array_greater(arr1, arr2):
    return np.any(np.greater(arr1, arr2))


class TestTraining(unittest.TestCase):

    def test_sleep_increase(self):
        pass

    def test_CPT_increase(self):
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
            # or True
        )
        # low_ses_student.display_original_cpts()
        # low_ses_student.display_cpts()
        low_ses_student.plot_memory()
        high_ses_student.plot_memory()



if __name__ == "__main__":
    unittest.main()
