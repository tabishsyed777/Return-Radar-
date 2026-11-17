import unittest
import numpy as np
from radar import generate, validate, split, pipeline, FEATURES, choose_threshold, cost, run, explain

class RadarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.data=generate(2500)

    def test_reproducible_data(self):
        self.assertTrue(generate(500).equals(generate(500)))

    def test_rejects_post_purchase_feature(self):
        bad=self.data.assign(refund_amount=99)
        with self.assertRaisesRegex(ValueError,'schema'): validate(bad)

    def test_rejects_premature_labels(self):
        bad=self.data.copy()
        bad['label_available_at']=bad.ordered_at
        with self.assertRaisesRegex(ValueError,'mature'): validate(bad)

    def test_temporal_separation_and_maturation(self):
        a,b,c=split(self.data)
        self.assertLess(a.label_available_at.max(),b.ordered_at.min())
        self.assertLess(b.label_available_at.max(),c.ordered_at.min())
        self.assertFalse(set(a.index)&set(b.index))
        self.assertFalse(set(b.index)&set(c.index))
        self.assertLess(len(a)+len(b)+len(c),len(self.data))

    def test_training_preprocessing_not_fitted_on_future(self):
        train,_,_=split(self.data)
        model=pipeline().fit(train[FEATURES],train.returned)
        stats=model['features'].named_transformers_['numeric']['impute'].statistics_
        np.testing.assert_allclose(stats,train[['price','discount','quantity','customer_tenure_days']].median())

    def test_unknown_category_and_missing_numeric(self):
        train,_,test=split(self.data)
        model=pipeline().fit(train[FEATURES],train.returned)
        sample=test.iloc[[0]].copy()
        sample['category']='new category'
        sample['price']=np.nan
        p=model.predict_proba(sample[FEATURES])[0,1]
        self.assertTrue(0<=p<=1)

    def test_threshold_minimizes_validation_cost(self):
        y=np.array([0,0,1,1]); p=np.array([.05,.1,.7,.9])
        t=choose_threshold(y,p)
        self.assertEqual(cost(y,p,t),0)

    def test_exact_explanation(self):
        train,_,test=split(self.data)
        model=pipeline().fit(train[FEATURES],train.returned)
        e=explain(model,test.iloc[[0]])
        self.assertAlmostEqual(1/(1+np.exp(-e['sum_log_odds'])),e['probability'],places=10)

    def test_pipeline_probability_and_reporting(self):
        _,report,predictions=run(self.data)
        self.assertTrue(predictions.return_probability.between(0,1).all())
        self.assertEqual(report['rows']['test'],len(predictions))
        self.assertEqual(sum(report['model']['confusion_matrix_tn_fp_fn_tp']),len(predictions))
        self.assertIn('training_prior_baseline',report)

    def test_holdout_labels_do_not_change_training_or_threshold(self):
        _, first, predictions_a=run(self.data)
        changed=self.data.copy()
        test_indices=split(changed)[2].index
        changed.loc[test_indices,'returned']=1-changed.loc[test_indices,'returned']
        _, second, predictions_b=run(changed)
        self.assertEqual(first['threshold_from_validation'],second['threshold_from_validation'])
        np.testing.assert_allclose(predictions_a.return_probability,predictions_b.return_probability)

    def test_single_class_rejected(self):
        bad=self.data.copy(); bad['returned']=0
        with self.assertRaisesRegex(ValueError,'both label'): split(bad)

    def test_bad_discount_rejected(self):
        bad=self.data.copy();bad.loc[0,'discount']=1.5
        with self.assertRaises(ValueError): validate(bad)

if __name__=='__main__': unittest.main()
