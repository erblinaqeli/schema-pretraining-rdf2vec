"""Tests for finetune init_strategy and init_relations hyperparameters."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from _maschine_init import (  # noqa: E402
    INIT_STRATEGIES,
    AllInitFallbackStats,
    apply_maschine_initialization,
    build_finetune_init_stats_payload,
    compute_finetune_init_stats,
    compute_instance_init_coverage,
    compute_relation_init_coverage,
    count_vocab_init_coverage,
    init_vector_from_class_roots_with_hops,
    normalize_init_strategy,
    restore_pretrained_vocab_rows,
    snapshot_class_initialized_entity_vectors,
    user_facing_init_strategy,
)
from _evaluate import build_eval_coverage_payload  # noqa: E402
from _pipeline import finetune_init_cli_args  # noqa: E402
from train import apply_synthetic_overrides, config_slug, extract_slug_params, load_default_flat, parse_args, TrainArgs  # noqa: E402
from _word2vec import InstanceAnchoringCallback  # noqa: E402
from _walks import nt_term  # noqa: E402

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
RDFS_DOMAIN = "http://www.w3.org/2000/01/rdf-schema#domain"
RDFS_RANGE = "http://www.w3.org/2000/01/rdf-schema#range"


class TestNormalizeInitStrategy(unittest.TestCase):
    def test_user_facing_names_map_to_internal(self) -> None:
        self.assertEqual(normalize_init_strategy("most_specific"), "most_specific")
        self.assertEqual(normalize_init_strategy("all_init"), "all_init")
        self.assertEqual(normalize_init_strategy("average_hier"), "ancestor_mean")
        self.assertEqual(normalize_init_strategy("weight_hierarchy"), "ancestor_weighted")

    def test_legacy_names_accepted(self) -> None:
        self.assertEqual(normalize_init_strategy("ancestor_mean"), "ancestor_mean")
        self.assertEqual(normalize_init_strategy("ancestor_weighted"), "ancestor_weighted")

    def test_user_facing_round_trip(self) -> None:
        for name in sorted(INIT_STRATEGIES):
            self.assertEqual(user_facing_init_strategy(name), name)
        self.assertEqual(user_facing_init_strategy("ancestor_mean"), "average_hier")
        self.assertEqual(user_facing_init_strategy("ancestor_weighted"), "weight_hierarchy")

    def test_unknown_strategy_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_init_strategy("invalid_strategy")


class TestFinetuneInitCliArgs(unittest.TestCase):
    def test_defaults(self) -> None:
        self.assertEqual(
            finetune_init_cli_args({}),
            [
                "--init-strategy",
                "most_specific",
                "--init-relations",
                "--initialization-noise",
                "0.0",
                "--anchor-regularization",
                "0.0",
            ],
        )

    def test_custom_values(self) -> None:
        self.assertEqual(
            finetune_init_cli_args(
                {"init_strategy": "average_hier", "init_relations": False},
            ),
            [
                "--init-strategy",
                "average_hier",
                "--no-init-relations",
                "--initialization-noise",
                "0.0",
                "--anchor-regularization",
                "0.0",
            ],
        )

    def test_initialization_noise(self) -> None:
        self.assertEqual(
            finetune_init_cli_args({"initialization_noise": 0.1}),
            [
                "--init-strategy",
                "most_specific",
                "--init-relations",
                "--initialization-noise",
                "0.1",
                "--anchor-regularization",
                "0.0",
            ],
        )

    def test_anchor_regularization(self) -> None:
        self.assertEqual(
            finetune_init_cli_args({"anchor_regularization": 0.05}),
            [
                "--init-strategy",
                "most_specific",
                "--init-relations",
                "--initialization-noise",
                "0.0",
                "--anchor-regularization",
                "0.05",
            ],
        )


class TestApplyMaschineInitRelations(unittest.TestCase):
    def _make_model_and_vectors(self, tmp_dir: Path) -> tuple[Word2Vec, dict[str, np.ndarray], Path]:
        inst = nt_term("I_tc07_1")
        cls = nt_term("C_tc07_1")
        pred = nt_term("P_tc07_1")

        model = Word2Vec(
            sentences=[[inst, pred]],
            vector_size=4,
            min_count=1,
            window=1,
            sg=1,
            epochs=1,
        )
        stage1_vectors = {
            cls: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            # Stage-1 pretrain row keyed by inner IRI (finetune token uses nt_term wrapper).
            "P_tc07_1": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        }

        ontology = tmp_dir / "ontology.nt"
        ontology.write_text(
            f"{inst} <{RDF_TYPE}> {cls} .\n",
            encoding="utf-8",
        )
        return model, stage1_vectors, ontology

    def test_init_relations_true_copies_predicate_rows(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            model, stage1_vectors, ontology = self._make_model_and_vectors(Path(tmp))
            pred = nt_term("P_tc07_1")

            new_token_init: dict[str, str] = {}
            _n_ent, n_rel, _n_rand, _stats = apply_maschine_initialization(
                model,
                stage1_vectors,
                ontology,
                init_relations=True,
                new_token_init=new_token_init,
            )

            self.assertEqual(n_rel, 1)
            np.testing.assert_allclose(model.wv[pred], stage1_vectors["P_tc07_1"])
            self.assertEqual(new_token_init[pred], "maschine_relation_copy")

    def test_init_relations_false_skips_predicate_rows(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            model, stage1_vectors, ontology = self._make_model_and_vectors(Path(tmp))
            pred = nt_term("P_tc07_1")
            before = model.wv[pred].copy()

            new_token_init: dict[str, str] = {}
            _n_ent, n_rel, n_rand, _stats = apply_maschine_initialization(
                model,
                stage1_vectors,
                ontology,
                init_relations=False,
                new_token_init=new_token_init,
            )

            self.assertEqual(n_rel, 0)
            self.assertGreater(n_rand, 0)
            np.testing.assert_allclose(model.wv[pred], before)
            self.assertEqual(new_token_init[pred], "random_relation")


class TestInstanceAnchoring(unittest.TestCase):
    def test_alpha_one_restores_init_vectors(self) -> None:
        import tempfile

        inst = nt_term("I_tc07_1")
        cls = nt_term("C_tc07_1")
        pred = nt_term("P_tc07_1")

        with tempfile.TemporaryDirectory() as tmp:
            model = Word2Vec(
                sentences=[[inst, pred]],
                vector_size=4,
                min_count=1,
                window=1,
                sg=1,
                epochs=1,
            )
            stage1_vectors = {
                cls: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "P_tc07_1": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            }
            ontology = Path(tmp) / "ontology.nt"
            ontology.write_text(
                f"{inst} <{RDF_TYPE}> {cls} .\n",
                encoding="utf-8",
            )

            new_token_init: dict[str, str] = {}
            apply_maschine_initialization(
                model,
                stage1_vectors,
                ontology,
                new_token_init=new_token_init,
            )
            init_vectors = snapshot_class_initialized_entity_vectors(model, new_token_init)
            self.assertIn(inst, init_vectors)

            idx = model.wv.key_to_index[inst]
            model.wv.vectors[idx] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

            InstanceAnchoringCallback(
                init_vectors, alpha=1.0, total_epochs=2
            ).on_epoch_end(model)
            np.testing.assert_allclose(model.wv[inst], init_vectors[inst])

    def test_skips_anchoring_on_last_epoch(self) -> None:
        import tempfile

        inst = nt_term("I_tc07_1")
        cls = nt_term("C_tc07_1")
        pred = nt_term("P_tc07_1")

        with tempfile.TemporaryDirectory() as tmp:
            model = Word2Vec(
                sentences=[[inst, pred]],
                vector_size=4,
                min_count=1,
                window=1,
                sg=1,
                epochs=1,
            )
            stage1_vectors = {
                cls: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "P_tc07_1": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            }
            ontology = Path(tmp) / "ontology.nt"
            ontology.write_text(
                f"{inst} <{RDF_TYPE}> {cls} .\n",
                encoding="utf-8",
            )

            new_token_init: dict[str, str] = {}
            apply_maschine_initialization(
                model,
                stage1_vectors,
                ontology,
                new_token_init=new_token_init,
            )
            init_vectors = snapshot_class_initialized_entity_vectors(model, new_token_init)

            idx = model.wv.key_to_index[inst]
            perturbed = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            model.wv.vectors[idx] = perturbed

            cb = InstanceAnchoringCallback(init_vectors, alpha=1.0, total_epochs=1)
            cb.on_epoch_end(model)
            np.testing.assert_allclose(model.wv[inst], perturbed)


class TestInitializationNoise(unittest.TestCase):
    def _make_model_and_vectors(self, tmp_dir: Path) -> tuple[Word2Vec, dict[str, np.ndarray], Path]:
        inst = nt_term("I_tc07_1")
        cls = nt_term("C_tc07_1")
        pred = nt_term("P_tc07_1")

        model = Word2Vec(
            sentences=[[inst, pred]],
            vector_size=4,
            min_count=1,
            window=1,
            sg=1,
            epochs=1,
        )
        stage1_vectors = {
            cls: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "P_tc07_1": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        }

        ontology = tmp_dir / "ontology.nt"
        ontology.write_text(
            f"{inst} <{RDF_TYPE}> {cls} .\n",
            encoding="utf-8",
        )
        return model, stage1_vectors, ontology

    def test_zero_noise_preserves_exact_vectors(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            model, stage1_vectors, ontology = self._make_model_and_vectors(Path(tmp))
            inst = nt_term("I_tc07_1")
            pred = nt_term("P_tc07_1")

            apply_maschine_initialization(
                model,
                stage1_vectors,
                ontology,
                initialization_noise=0.0,
            )

            np.testing.assert_allclose(model.wv[inst], stage1_vectors[nt_term("C_tc07_1")])
            np.testing.assert_allclose(model.wv[pred], stage1_vectors["P_tc07_1"])
            if model.syn1neg is not None:
                np.testing.assert_allclose(model.syn1neg[model.wv.key_to_index[inst]], model.wv[inst])
                np.testing.assert_allclose(model.syn1neg[model.wv.key_to_index[pred]], model.wv[pred])

    def test_nonzero_noise_perturbs_vectors(self) -> None:
        import tempfile

        np.random.seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            model, stage1_vectors, ontology = self._make_model_and_vectors(Path(tmp))
            inst = nt_term("I_tc07_1")
            pred = nt_term("P_tc07_1")
            expected_inst = stage1_vectors[nt_term("C_tc07_1")]
            expected_pred = stage1_vectors["P_tc07_1"]

            apply_maschine_initialization(
                model,
                stage1_vectors,
                ontology,
                initialization_noise=0.5,
            )

            self.assertFalse(np.allclose(model.wv[inst], expected_inst))
            self.assertFalse(np.allclose(model.wv[pred], expected_pred))
            if model.syn1neg is not None:
                np.testing.assert_allclose(model.syn1neg[model.wv.key_to_index[inst]], model.wv[inst])
                np.testing.assert_allclose(model.syn1neg[model.wv.key_to_index[pred]], model.wv[pred])


class TestInstanceInitLabels(unittest.TestCase):
    def _run_init(
        self,
        tmp_dir: Path,
        *,
        stage1_vectors: dict[str, np.ndarray],
        ontology_text: str,
        sentences: list[list[str]],
    ) -> dict[str, str]:
        ontology = tmp_dir / "ontology.nt"
        ontology.write_text(ontology_text, encoding="utf-8")
        model = Word2Vec(
            sentences=sentences,
            vector_size=4,
            min_count=1,
            window=1,
            sg=1,
            epochs=1,
        )
        new_token_init: dict[str, str] = {}
        apply_maschine_initialization(
            model,
            stage1_vectors,
            ontology,
            init_relations=False,
            new_token_init=new_token_init,
        )
        return new_token_init

    def test_instance_with_class_and_pretrain_vector_gets_maschine_label(self) -> None:
        import tempfile

        inst = nt_term("I_tc01_1")
        cls = nt_term("C_tc01_1")
        with tempfile.TemporaryDirectory() as tmp:
            labels = self._run_init(
                Path(tmp),
                stage1_vectors={cls: np.ones(4, dtype=np.float32)},
                ontology_text=f"{inst} <{RDF_TYPE}> {cls} .\n",
                sentences=[[inst]],
            )
            self.assertEqual(labels[inst], "maschine_class_mean")

    def test_instance_with_class_but_no_pretrain_vector_gets_random_no_pretrained_class(self) -> None:
        import tempfile

        inst = nt_term("I_tc01_2")
        cls = nt_term("C_tc01_2")
        with tempfile.TemporaryDirectory() as tmp:
            labels = self._run_init(
                Path(tmp),
                stage1_vectors={},
                ontology_text=f"{inst} <{RDF_TYPE}> {cls} .\n",
                sentences=[[inst]],
            )
            self.assertEqual(labels[inst], "random_no_pretrained_class")

    def test_instance_without_class_gets_random_no_class_mapping(self) -> None:
        import tempfile

        inst = nt_term("I_tc01_3")
        with tempfile.TemporaryDirectory() as tmp:
            labels = self._run_init(
                Path(tmp),
                stage1_vectors={},
                ontology_text="",
                sentences=[[inst]],
            )
            self.assertEqual(labels[inst], "random_no_class_mapping")


class TestInstanceInitCoverage(unittest.TestCase):
    def _make_setup(
        self,
        tmp_dir: Path,
        *,
        ontology_text: str,
        sentences: list[list[str]],
        stage1_vectors: dict[str, np.ndarray],
    ) -> tuple[Word2Vec, dict[str, np.ndarray], dict[str, str], Path]:
        ontology = tmp_dir / "ontology.nt"
        ontology.write_text(ontology_text, encoding="utf-8")
        model = Word2Vec(
            sentences=sentences,
            vector_size=4,
            min_count=1,
            window=1,
            sg=1,
            epochs=1,
        )
        new_token_init: dict[str, str] = {}
        apply_maschine_initialization(
            model,
            stage1_vectors,
            ontology,
            init_relations=False,
            new_token_init=new_token_init,
        )
        return model, stage1_vectors, new_token_init, ontology

    def test_coverage_classifies_all_outcomes(self) -> None:
        import tempfile

        inst_ok = nt_term("I_tc01_10")
        inst_missing = nt_term("I_tc01_11")
        inst_no_type = nt_term("I_tc01_12")
        inst_absent = nt_term("I_tc01_13")
        cls_ok = nt_term("C_tc01_10")
        cls_missing = nt_term("C_tc01_11")

        with tempfile.TemporaryDirectory() as tmp:
            ontology_text = (
                f"{inst_ok} <{RDF_TYPE}> {cls_ok} .\n"
                f"{inst_missing} <{RDF_TYPE}> {cls_missing} .\n"
            )
            model, stage1_vectors, new_token_init, ontology = self._make_setup(
                Path(tmp),
                ontology_text=ontology_text,
                sentences=[[inst_ok], [inst_missing], [inst_no_type]],
                stage1_vectors={cls_ok: np.ones(4, dtype=np.float32)},
            )

            stats = compute_instance_init_coverage(
                ontology,
                model,
                stage1_vectors,
                new_token_init,
            )

            self.assertEqual(stats.total_instances, 2)
            self.assertEqual(stats.with_class_mapping, 2)
            self.assertEqual(stats.without_class_mapping, 0)
            self.assertEqual(stats.initialized_from_class, 1)
            self.assertEqual(stats.random_no_pretrained_class, 1)
            self.assertEqual(stats.random_no_class_mapping, 0)
            self.assertEqual(stats.not_in_finetune_vocab, 0)

    def test_coverage_counts_not_in_vocab(self) -> None:
        import tempfile

        inst_in = nt_term("I_tc01_20")
        inst_out = nt_term("I_tc01_21")
        cls = nt_term("C_tc01_20")

        with tempfile.TemporaryDirectory() as tmp:
            ontology_text = (
                f"{inst_in} <{RDF_TYPE}> {cls} .\n"
                f"{inst_out} <{RDF_TYPE}> {cls} .\n"
            )
            model, stage1_vectors, new_token_init, ontology = self._make_setup(
                Path(tmp),
                ontology_text=ontology_text,
                sentences=[[inst_in]],
                stage1_vectors={cls: np.ones(4, dtype=np.float32)},
            )

            stats = compute_instance_init_coverage(
                ontology,
                model,
                stage1_vectors,
                new_token_init,
            )

            self.assertEqual(stats.total_instances, 2)
            self.assertEqual(stats.initialized_from_class, 1)
            self.assertEqual(stats.not_in_finetune_vocab, 1)

    def test_coverage_handles_bare_walk_tokens(self) -> None:
        import tempfile

        inst = "I_tc01_30"
        cls = "C_tc01_30"
        inst_nt = nt_term(inst)
        cls_nt = nt_term(cls)

        with tempfile.TemporaryDirectory() as tmp:
            ontology_text = f"{inst_nt} <{RDF_TYPE}> {cls_nt} .\n"
            model, stage1_vectors, new_token_init, ontology = self._make_setup(
                Path(tmp),
                ontology_text=ontology_text,
                sentences=[[inst]],
                stage1_vectors={cls: np.ones(4, dtype=np.float32)},
            )

            stats = compute_instance_init_coverage(
                ontology,
                model,
                stage1_vectors,
                new_token_init,
            )

            self.assertEqual(stats.total_instances, 1)
            self.assertEqual(stats.initialized_from_class, 1)
            self.assertEqual(stats.not_in_finetune_vocab, 0)


class TestCountVocabInitCoverage(unittest.TestCase):
    def test_counts_initialized_entities_and_relations(self) -> None:
        inst = nt_term("I_tc01_60")
        rel = nt_term("P_tc01_60")
        inst_rand = nt_term("I_tc01_61")
        word2idx = {inst: 0, rel: 1, inst_rand: 2}
        token_init_labels = {
            inst: "maschine_class_mean",
            rel: "maschine_relation_copy",
            inst_rand: "random",
        }

        counts = count_vocab_init_coverage(word2idx, token_init_labels)

        self.assertEqual(counts["initialized_instances"], 1)
        self.assertEqual(counts["initialized_relations"], 1)

    def test_counts_instances_without_angle_brackets(self) -> None:
        inst = "I_tc01_60"
        word2idx = {inst: 0}
        token_init_labels = {inst: "maschine_class_mean"}

        counts = count_vocab_init_coverage(word2idx, token_init_labels)

        self.assertEqual(counts["initialized_instances"], 1)
        self.assertEqual(counts["maschine_initialized"], 1)


class TestRestorePretrainedVocabRows(unittest.TestCase):
    def test_rebuild_vocab_drops_pretrain_only_tokens(self) -> None:
        inst = nt_term("I_tc01_50")
        rel = nt_term("P_tc01_50")
        cls_pretrain_only = nt_term("C_tc01_pretrain_only")

        pretrain_model = Word2Vec(
            sentences=[[cls_pretrain_only, rel]],
            vector_size=4,
            min_count=1,
            window=1,
            sg=1,
            epochs=1,
        )
        stage1_vectors = {
            w: np.asarray(pretrain_model.wv[w], dtype=np.float32).copy()
            for w in pretrain_model.wv.index_to_key
        }
        self.assertEqual(len(stage1_vectors), 2)

        finetune_model = Word2Vec(
            vector_size=pretrain_model.vector_size,
            min_count=pretrain_model.min_count,
            window=pretrain_model.window,
            sg=pretrain_model.sg,
            epochs=1,
        )
        finetune_model.build_vocab([[inst, rel]])
        self.assertEqual(set(finetune_model.wv.index_to_key), {inst, rel})
        pretrain_model = finetune_model

        new_token_init: dict[str, str] = {}
        restored = restore_pretrained_vocab_rows(
            pretrain_model,
            stage1_vectors,
            new_token_init=new_token_init,
        )

        self.assertEqual(restored, 1)
        self.assertTrue(np.allclose(pretrain_model.wv[rel], stage1_vectors[rel]))
        self.assertEqual(new_token_init.get(rel), "stage1_pretrained")
        self.assertNotIn(cls_pretrain_only, pretrain_model.wv)


class TestFinetuneInitStatsBreakdown(unittest.TestCase):
    def test_separates_instance_and_class_tokens(self) -> None:
        import tempfile

        inst = nt_term("I_tc01_30")
        cls = nt_term("C_tc01_30")
        with tempfile.TemporaryDirectory() as tmp:
            ontology = Path(tmp) / "ontology.nt"
            ontology.write_text(f"{inst} <{RDF_TYPE}> {cls} .\n", encoding="utf-8")
            model = Word2Vec(
                sentences=[[inst, cls]],
                vector_size=4,
                min_count=1,
                window=1,
                sg=1,
                epochs=1,
            )
            stage1_vectors = {cls: np.ones(4, dtype=np.float32)}
            new_token_init: dict[str, str] = {}
            apply_maschine_initialization(
                model,
                stage1_vectors,
                ontology,
                init_relations=False,
                new_token_init=new_token_init,
            )

            stats = compute_finetune_init_stats(model, stage1_vectors, new_token_init)

            self.assertEqual(stats.instances.from_protograph, 1)
            self.assertEqual(stats.instances.random, 0)
            self.assertEqual(stats.classes.from_protograph, 0)
            self.assertEqual(stats.classes.random, 0)


class TestFinetuneInitStatsPayload(TestInstanceInitCoverage):
    def _make_setup_with_relations(
        self,
        tmp_dir: Path,
        *,
        ontology_text: str,
        sentences: list[list[str]],
        stage1_vectors: dict[str, np.ndarray],
    ) -> tuple[Word2Vec, dict[str, np.ndarray], dict[str, str], Path]:
        ontology = tmp_dir / "ontology.nt"
        ontology.write_text(ontology_text, encoding="utf-8")
        model = Word2Vec(
            sentences=sentences,
            vector_size=4,
            min_count=1,
            window=1,
            sg=1,
            epochs=1,
        )
        new_token_init: dict[str, str] = {}
        apply_maschine_initialization(
            model,
            stage1_vectors,
            ontology,
            init_relations=True,
            new_token_init=new_token_init,
        )
        return model, stage1_vectors, new_token_init, ontology

    def test_simplified_payload_uses_instance_coverage(self) -> None:
        import tempfile

        inst_in = nt_term("I_tc01_40")
        inst_out = nt_term("I_tc01_41")
        rel_in = nt_term("P_tc01_40")
        rel_out = nt_term("P_tc01_41")
        cls = nt_term("C_tc01_40")

        with tempfile.TemporaryDirectory() as tmp:
            ontology_text = (
                f"{inst_in} <{RDF_TYPE}> {cls} .\n"
                f"{inst_out} <{RDF_TYPE}> {cls} .\n"
                f"{rel_in} <{RDFS_DOMAIN}> {cls} .\n"
                f"{rel_out} <{RDFS_RANGE}> {cls} .\n"
            )
            model, stage1_vectors, new_token_init, ontology = self._make_setup_with_relations(
                Path(tmp),
                ontology_text=ontology_text,
                sentences=[[inst_in], [rel_in]],
                stage1_vectors={
                    cls: np.ones(4, dtype=np.float32),
                    rel_in[1:-1]: np.full(4, 2.0, dtype=np.float32),
                },
            )

            init_stats = compute_finetune_init_stats(model, stage1_vectors, new_token_init)
            instance_coverage = compute_instance_init_coverage(
                ontology,
                model,
                stage1_vectors,
                new_token_init,
            )
            relation_coverage = compute_relation_init_coverage(
                ontology,
                model,
                stage1_vectors,
                new_token_init,
            )
            payload = build_finetune_init_stats_payload(
                init_stats,
                instance_coverage=instance_coverage,
                relation_coverage=relation_coverage,
            )

            self.assertEqual(payload["vocabulary_size"], 1)
            self.assertEqual(payload["instances"]["total_graph_instances"], 2)
            self.assertEqual(payload["instances"]["init_from_protograph"], 1)
            self.assertEqual(payload["instances"]["init_random"], 0)
            self.assertEqual(payload["instances"]["not_in_walk_corpus"], 1)
            self.assertEqual(payload["relations"]["total_graph_relations"], 2)
            self.assertEqual(payload["relations"]["init_from_protograph"], 1)
            self.assertEqual(payload["relations"]["not_initialized"], 1)


class TestAllInitStrategy(unittest.TestCase):
    def test_fallback_resolves_parent_vector(self) -> None:
        import tempfile

        inst = nt_term("I_tc01_child")
        cls_child = nt_term("C_tc01_child")
        cls_parent = nt_term("C_tc01_parent")
        parents = {"C_tc01_child": {"C_tc01_parent"}}
        stage1_vectors = {
            cls_parent: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        vec, hops = init_vector_from_class_roots_with_hops(
            ["C_tc01_child"],
            parents,
            stage1_vectors,
            strategy="all_init",
        )
        self.assertIsNotNone(vec)
        np.testing.assert_allclose(vec, stage1_vectors[cls_parent])
        self.assertEqual(hops, 1)

    def test_direct_most_specific_has_zero_hops(self) -> None:
        parents: dict[str, set[str]] = {}
        cls = nt_term("C_tc01_1")
        stage1_vectors = {cls: np.ones(4, dtype=np.float32)}
        vec, hops = init_vector_from_class_roots_with_hops(
            ["C_tc01_1"],
            parents,
            stage1_vectors,
            strategy="all_init",
        )
        self.assertIsNotNone(vec)
        self.assertEqual(hops, 0)

    def test_no_ancestor_in_pretrain_returns_none(self) -> None:
        parents = {"C_tc01_child": {"C_tc01_parent"}}
        vec, hops = init_vector_from_class_roots_with_hops(
            ["C_tc01_child"],
            parents,
            {},
            strategy="all_init",
        )
        self.assertIsNone(vec)
        self.assertEqual(hops, 0)

    def test_apply_maschine_all_init_with_fallback(self) -> None:
        import tempfile

        inst = nt_term("I_tc01_fb")
        cls_child = nt_term("C_tc01_fb_child")
        cls_parent = nt_term("C_tc01_fb_parent")
        with tempfile.TemporaryDirectory() as tmp:
            ontology = Path(tmp) / "ontology.nt"
            ontology.write_text(
                f"{inst} <{RDF_TYPE}> {cls_child} .\n"
                f"{cls_child} <{RDFS_SUBCLASS}> {cls_parent} .\n",
                encoding="utf-8",
            )
            model = Word2Vec(
                sentences=[[inst]],
                vector_size=4,
                min_count=1,
                window=1,
                sg=1,
                epochs=1,
            )
            stage1_vectors = {
                cls_parent: np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32),
            }
            new_token_init: dict[str, str] = {}
            n_ent, _n_rel, n_rand, stats = apply_maschine_initialization(
                model,
                stage1_vectors,
                ontology,
                strategy="all_init",
                init_relations=False,
                new_token_init=new_token_init,
            )
            self.assertEqual(n_ent, 1)
            self.assertEqual(n_rand, 0)
            self.assertIsNotNone(stats)
            assert stats is not None
            self.assertEqual(stats.instances_with_fallback, 1)
            self.assertEqual(stats.max_hops, 1)
            self.assertEqual(stats.avg_hops, 1.0)
            self.assertEqual(new_token_init[inst], "maschine_class_mean")
            np.testing.assert_allclose(model.wv[inst], stage1_vectors[cls_parent])

    def test_all_init_stats_aggregation(self) -> None:
        stats = AllInitFallbackStats()
        stats.record(0)
        stats.record(2)
        self.assertEqual(stats.instances_with_fallback, 1)
        self.assertEqual(stats.max_hops, 2)
        self.assertEqual(stats.avg_hops, 1.0)

    def test_finetune_init_stats_payload_includes_all_init_fallback(self) -> None:
        import tempfile

        inst_direct = nt_term("I_tc01_direct")
        inst_fb = nt_term("I_tc01_fb")
        cls_direct = nt_term("C_tc01_direct")
        cls_child = nt_term("C_tc01_child")
        cls_parent = nt_term("C_tc01_parent")
        with tempfile.TemporaryDirectory() as tmp:
            ontology = Path(tmp) / "ontology.nt"
            ontology.write_text(
                f"{inst_direct} <{RDF_TYPE}> {cls_direct} .\n"
                f"{inst_fb} <{RDF_TYPE}> {cls_child} .\n"
                f"{cls_child} <{RDFS_SUBCLASS}> {cls_parent} .\n",
                encoding="utf-8",
            )
            model = Word2Vec(
                sentences=[[inst_direct], [inst_fb]],
                vector_size=4,
                min_count=1,
                window=1,
                sg=1,
                epochs=1,
            )
            stage1_vectors = {
                cls_direct: np.ones(4, dtype=np.float32),
                cls_parent: np.full(4, 2.0, dtype=np.float32),
            }
            new_token_init: dict[str, str] = {}
            _n_ent, _n_rel, _n_rand, all_init_stats = apply_maschine_initialization(
                model,
                stage1_vectors,
                ontology,
                strategy="all_init",
                init_relations=False,
                new_token_init=new_token_init,
            )
            init_stats = compute_finetune_init_stats(model, stage1_vectors, new_token_init)
            payload = build_finetune_init_stats_payload(
                init_stats,
                init_strategy="all_init",
                all_init_fallback=all_init_stats,
            )
            self.assertEqual(payload["init_strategy"], "all_init")
            fb = payload["all_init_fallback"]
            assert isinstance(fb, dict)
            self.assertEqual(fb["instances_with_fallback"], 1)
            self.assertEqual(fb["max_hops"], 1)
            self.assertEqual(fb["avg_hops"], 0.5)

    def test_eval_coverage_payload_includes_all_init_fallback(self) -> None:
        payload = build_eval_coverage_payload(
            {
                "test_path": "/tmp/test.txt",
                "train_path": "/tmp/train.txt",
                "checkpoint_path": "/tmp/ckpt.pt",
                "init_labels_available": True,
                "init_strategy": "all_init",
                "all_init_fallback": {
                    "instances_with_fallback": 3,
                    "max_hops": 2,
                    "avg_hops": 0.75,
                },
                "vocab_coverage": {
                    "total": 10,
                    "maschine_initialized": 8,
                    "stage1_pretrained": 0,
                    "initialized_instances": 8,
                    "initialized_relations": 0,
                    "random_initialized": 2,
                    "unknown": 0,
                },
                "n_test": 4,
                "n_test_maschine_initialized": 3,
            }
        )
        self.assertEqual(payload["init_strategy"], "all_init")
        self.assertEqual(
            payload["all_init_fallback"],
            {
                "instances_with_fallback": 3,
                "max_hops": 2,
                "avg_hops": 0.75,
            },
        )


class TestTrainInitStrategyCli(unittest.TestCase):
    def test_parse_args_init_strategy(self) -> None:
        args, _remainder = parse_args(
            ["--dataset", "synthetic", "--train-mode", "p2", "--init-strategy", "all_init"]
        )
        self.assertEqual(args.init_strategy, "all_init")

    def test_apply_synthetic_overrides_init_strategy(self) -> None:
        flat = load_default_flat("p2")
        args = TrainArgs(
            datasets=["synthetic"],
            tcs=None,
            train_modes=["p2"],
            config_name=None,
            config_path=None,
            timestamp=None,
            pretrain_epochs=None,
            finetune_epochs=None,
            epochs=None,
            finetune_lr=None,
            pretrain_lr=None,
            dim=None,
            window=None,
            workers=None,
            seed=None,
            negative=None,
            min_count=None,
            initialization_noise=None,
            anchor_regularization=None,
            init_strategy="all_init",
            pretrain_walks_per_entity=None,
            pretrain_depth=None,
            finetune_walks_per_entity=None,
            finetune_depth=None,
            instance_walks=None,
            pretrain_walks_p1=None,
            pretrain_walks_p2=None,
            ontology=None,
            instance_types=None,
            precomputed_instance_walks=None,
            no_eval=False,
            dry_run=False,
        )
        out = apply_synthetic_overrides(flat, "p2", args)
        self.assertEqual(out["finetune_word2vec"]["init_strategy"], "all_init")

    def test_all_init_config_slug(self) -> None:
        flat = load_default_flat("p2")
        flat["finetune_word2vec"]["init_strategy"] = "all_init"
        defaults = extract_slug_params(load_default_flat("p2"), "p2")
        current = extract_slug_params(flat, "p2")
        slug = config_slug(current, defaults, explicit_name=None)
        self.assertEqual(slug, "init_strategy_all_init")


if __name__ == "__main__":
    unittest.main()
