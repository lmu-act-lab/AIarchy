from student import Student
import unittest
import numpy as np


low_ses_student = Student(
    {"SES": 0}, weights={"health": 0.33, "social": 0.33, "grades": 0.33}
)
high_ses_student = Student(
    {"SES": 2}, weights={"health": 0.33, "social": 0.33, "grades": 0.33}
)
high_ses_student.train(1000)
low_ses_student.train(1000)


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
        low_ses_student.write_cpds_to_csv(
            low_ses_student.get_cpts(), "trained low_ses", "weight_changing"
        )
        low_ses_student.write_delta_cpd_to_csv(
            low_ses_student.get_cpts(), "delta low_ses", "weight_changing"
        )
        high_ses_student.write_cpds_to_csv(
            high_ses_student.get_cpts(), "trained high_ses", "weight_changing"
        )
        high_ses_student.write_delta_cpd_to_csv(
            high_ses_student.get_cpts(), "delta high_ses", "weight_changing"
        )
        low_ses_student.display_weights()
        high_ses_student.display_weights()


if __name__ == "__main__":
    unittest.main()
