import unittest

import omnivoice_language as lang


class DetectLanguageRequestTests(unittest.TestCase):
    def test_switch_to_phrasing(self):
        self.assertEqual(lang.detect_language_request("can you switch to Spanish?"), ("es", "Spanish"))
        self.assertEqual(lang.detect_language_request("let's switch to French"), ("fr", "French"))

    def test_respond_in_phrasing(self):
        self.assertEqual(lang.detect_language_request("please respond in Japanese"), ("ja", "Japanese"))
        self.assertEqual(lang.detect_language_request("answer in German from now on"), ("de", "German"))

    def test_speak_direct_phrasing(self):
        self.assertEqual(lang.detect_language_request("can you speak Italian"), ("it", "Italian"))
        self.assertEqual(lang.detect_language_request("talk to me in Portuguese"), ("pt", "Portuguese"))

    def test_native_names_and_imperatives(self):
        self.assertEqual(lang.detect_language_request("switch to español"), ("es", "Spanish"))
        self.assertEqual(lang.detect_language_request("habla español por favor"), ("es", "Spanish"))
        self.assertEqual(lang.detect_language_request("parle français"), ("fr", "French"))

    def test_no_false_positive_on_casual_mention(self):
        self.assertIsNone(lang.detect_language_request("I learned French in high school"))
        self.assertIsNone(lang.detect_language_request("my family speaks three languages"))
        self.assertIsNone(lang.detect_language_request("how do you say hello"))

    def test_unknown_language_returns_none(self):
        self.assertIsNone(lang.detect_language_request("switch to Klingon"))

    def test_empty(self):
        self.assertIsNone(lang.detect_language_request(""))
        self.assertIsNone(lang.detect_language_request(None))

    def test_language_name_helper(self):
        self.assertEqual(lang.language_name("es"), "Spanish")
        self.assertEqual(lang.language_name("en"), "English")


if __name__ == "__main__":
    unittest.main()
