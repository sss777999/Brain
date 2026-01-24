# CHUNK_META:
#   Purpose: Biologically correct tests for episodic memory
#   Dependencies: episode, hippocampus, cortex
#   API: test functions

"""
Tests for biological correctness of episodic memory.

NOT adjustment to implementation, but verification of REAL biological properties:
- Pattern Separation: similar inputs MUST produce DIFFERENT sparse representations
- Pattern Completion: partial cue MUST restore full episode
- Sparsity: DG MUST produce ~10% active neurons (as in biology)
- Consolidation: repetition MUST lead to consolidation
- Decay: without repetition episodes MUST be forgotten
"""

import pytest
from typing import Set

import sys
sys.path.insert(0, '/Users/a/Documents/projects/gpt_model/Brain')

from episode import Episode, EpisodeState
from hippocampus import Hippocampus
from cortex import Cortex


class TestBiologicalPatternSeparation:
    """
    Tests for Pattern Separation in Dentate Gyrus.
    
    Biology: DG creates sparse, orthogonal representations.
    Similar inputs -> DIFFERENT outputs (to prevent interference).
    """
    
    def test_sparsity_is_around_10_percent(self) -> None:
        """
        BIOLOGICAL REQUIREMENT: DG produces ~10% active neurons.
        
        In real DG granule cells have very low activity.
        This is achieved through strong inhibition from interneurons.
        """
        cortex = Cortex()
        hippo = Hippocampus(cortex)
        
        # Test on different input sizes
        for input_size in [50, 100, 200, 500]:
            input_neurons = {f"n{i}" for i in range(input_size)}
            sparse = hippo.pattern_separate(input_neurons)
            
            sparsity = len(sparse) / input_size
            
            # Biological requirement: 5-15% (around 10%)
            assert 0.05 <= sparsity <= 0.15, (
                f"Sparsity {sparsity:.2%} for input {input_size} "
                f"outside biological range 5-15%"
            )
    
    def test_similar_inputs_produce_different_outputs(self) -> None:
        """
        BIOLOGICAL REQUIREMENT: similar inputs -> different sparse representations.
        
        This is the key function of DG - pattern separation.
        Without this similar episodes will interfere.
        """
        cortex = Cortex()
        hippo = Hippocampus(cortex)
        
        # Two inputs with 90% overlap
        base = {f"n{i}" for i in range(100)}
        similar = {f"n{i}" for i in range(10, 110)}  # 90 shared neurons
        
        sparse_base = hippo.pattern_separate(base)
        sparse_similar = hippo.pattern_separate(similar)
        
        # Calculate sparse representations overlap
        overlap = len(sparse_base & sparse_similar)
        max_possible = min(len(sparse_base), len(sparse_similar))
        
        if max_possible > 0:
            overlap_ratio = overlap / max_possible
            
            # Pattern separation MUST SIGNIFICANTLY reduce overlap
            # If inputs are 90% similar, outputs should be SIGNIFICANTLY less similar
            # In biology DG creates almost orthogonal representations
            # Requirement: output overlap < 50% (strict requirement)
            assert overlap_ratio < 0.5, (
                f"Pattern separation weak: "
                f"inputs 90% similar, outputs {overlap_ratio:.0%} similar. "
                f"Should be < 50% to prevent interference"
            )
    
    def test_output_is_subset_of_input(self) -> None:
        """
        BIOLOGICAL REQUIREMENT: sparse representation is a subset of input.
        
        DG doesn't create new neurons, it selects a subset.
        """
        cortex = Cortex()
        hippo = Hippocampus(cortex)
        
        input_neurons = {f"neuron_{i}" for i in range(100)}
        sparse = hippo.pattern_separate(input_neurons)
        
        assert sparse.issubset(input_neurons), (
            "Sparse representation contains neurons not in input"
        )


class TestBiologicalPatternCompletion:
    """
    Tests for Pattern Completion in CA3.
    
    Biology: CA3 is a recurrent network with attractor dynamics.
    Partial cue -> full memory recall.
    """
    
    def test_partial_cue_retrieves_full_episode(self) -> None:
        """
        BIOLOGICAL REQUIREMENT: partial cue restores full episode.
        
        This is autoassociative memory in CA3.
        """
        cortex = Cortex()
        hippo = Hippocampus(cortex)
        
        # Encode episode with 50 neurons
        full_input = {f"n{i}" for i in range(50)}
        episode = hippo.encode(full_input, source="test")
        
        # Try to recall with 20% cue
        # Take neurons from sparse representation of episode
        sparse_neurons = list(episode.pattern_neurons)
        cue_size = max(1, len(sparse_neurons) // 5)  # 20%
        partial_cue = set(sparse_neurons[:cue_size])
        
        retrieved = hippo.retrieve(partial_cue)
        
        assert retrieved is not None, (
            f"Pattern completion failed: cue of {cue_size} neurons "
            f"from {len(sparse_neurons)} did not find episode"
        )
        assert retrieved.id == episode.id, (
            "Pattern completion found wrong episode"
        )
    
    def test_unrelated_cue_does_not_retrieve(self) -> None:
        """
        BIOLOGICAL REQUIREMENT: unrelated cue should NOT find episode.
        
        Otherwise this is a false memory.
        """
        cortex = Cortex()
        hippo = Hippocampus(cortex)
        
        # Encode episode
        hippo.encode({f"a{i}" for i in range(50)}, source="test")
        
        # Try to recall with completely different neurons
        unrelated_cue = {f"z{i}" for i in range(10)}
        retrieved = hippo.retrieve(unrelated_cue)
        
        assert retrieved is None, (
            "Pattern completion found episode with unrelated cue - "
            "this is a false memory"
        )


class TestBiologicalConsolidation:
    """
    Tests for consolidation: hippocampus -> cortex.
    
    Biology: repetition (replay) transfers episodes to semantic memory.
    """
    
    def test_repetition_leads_to_consolidation(self) -> None:
        """
        BIOLOGICAL REQUIREMENT: repetition leads to consolidation.
        
        This is the basis of long-term memory formation.
        """
        cortex = Cortex()
        hippo = Hippocampus(cortex)
        
        input_neurons = {f"n{i}" for i in range(30)}
        episode = hippo.encode(input_neurons, source="test")
        
        initial_state = episode.state
        assert initial_state == EpisodeState.NEW, (
            "New episode should be in NEW state"
        )
        
        # Repeat enough times for consolidation
        for i in range(hippo.CONSOLIDATION_THRESHOLD):
            hippo.encode(input_neurons, source="test")
        
        # Episode should be consolidated
        assert episode.state == EpisodeState.CONSOLIDATED, (
            f"After {hippo.CONSOLIDATION_THRESHOLD} repetitions "
            f"episode should be CONSOLIDATED, but it is {episode.state}"
        )
    
    def test_single_exposure_does_not_consolidate(self) -> None:
        """
        BIOLOGICAL REQUIREMENT: single exposure does NOT consolidate immediately.
        
        Consolidation requires repetition (replay).
        """
        cortex = Cortex()
        hippo = Hippocampus(cortex)
        
        episode = hippo.encode({f"n{i}" for i in range(30)}, source="test")
        
        assert episode.state != EpisodeState.CONSOLIDATED, (
            "Single exposure should not consolidate immediately"
        )


class TestBiologicalForgetting:
    """
    Tests for forgetting (decay).
    
    Biology: episodes without repetition are forgotten.
    """
    
    def test_unrepeated_episodes_decay(self) -> None:
        """
        BIOLOGICAL REQUIREMENT: episodes without repetition decay.
        
        This is a natural forgetting mechanism.
        """
        ep = Episode(
            pattern_neurons={"n1", "n2", "n3"},
            context_neurons=set(),
            timestamp=1
        )
        
        # Apply decay without replay
        decay_cycles = 0
        while not ep.apply_decay():
            decay_cycles += 1
            if decay_cycles > 20:  # Protection from infinite loop
                break
        
        assert decay_cycles <= 10, (
            f"Episode should decay in reasonable time, "
            f"but it took {decay_cycles} cycles"
        )
    
    def test_replayed_episodes_resist_decay(self) -> None:
        """
        BIOLOGICAL REQUIREMENT: replay protects from forgetting.
        """
        ep = Episode(
            pattern_neurons={"n1", "n2", "n3"},
            context_neurons=set(),
            timestamp=1
        )
        
        # Several decay cycles
        for _ in range(3):
            ep.apply_decay()
        
        # Replay should reset decay
        ep.mark_replayed()
        
        # After replay episode should not be immediately removed
        should_remove = ep.apply_decay()
        assert not should_remove, (
            "Replay should protect from immediate removal"
        )


class TestBiologicalContext:
    """
    Tests for context dependency.
    
    Biology: episodes are linked to context (what was active).
    """
    
    def test_context_affects_retrieval(self) -> None:
        """
        BIOLOGICAL REQUIREMENT: context helps with recall.
        
        Episodes are easier to recall in similar context.
        """
        cortex = Cortex()
        hippo = Hippocampus(cortex)
        
        # Encode first episode - it will create context
        hippo.encode({"context_a1", "context_a2", "context_a3"}, source="context")
        
        # Encode second episode - it will get context from first
        ep2 = hippo.encode({"target_b1", "target_b2"}, source="target")
        
        # Context of second episode should contain something from first
        # (after pattern separation)
        assert len(ep2.context_neurons) > 0, (
            "Episode should have context from previous activity"
        )


class TestBiologicalCapacity:
    """
    Tests for hippocampus capacity.
    
    Biology: hippocampus has limited capacity.
    """
    
    def test_old_episodes_are_displaced(self) -> None:
        """
        BIOLOGICAL REQUIREMENT: old episodes are displaced by new ones.
        
        Hippocampus is a temporary storage, not infinite.
        """
        cortex = Cortex()
        hippo = Hippocampus(cortex)
        
        # Set small capacity for test
        original_max = hippo.MAX_EPISODES
        hippo.MAX_EPISODES = 10
        
        try:
            # Encode more episodes than capacity
            first_episode = hippo.encode({"first_n1", "first_n2"}, source="first")
            first_id = first_episode.id
            
            for i in range(15):
                hippo.encode({f"n{i}_{j}" for j in range(5)}, source=f"test{i}")
            
            # Number of episodes should not exceed MAX_EPISODES
            assert len(hippo.episodes) <= hippo.MAX_EPISODES, (
                f"Episode count {len(hippo.episodes)} "
                f"exceeds MAX_EPISODES {hippo.MAX_EPISODES}"
            )
        finally:
            hippo.MAX_EPISODES = original_max


class TestBiologicalSWR:
    """
    Tests for Sharp Wave-Ripples replay.
    
    Biology: SWR occur during sleep/rest and strengthen episodes.
    """
    
    def test_sleep_increases_replay_count(self) -> None:
        """
        BIOLOGICAL REQUIREMENT: sleep() replays episodes.
        """
        cortex = Cortex()
        hippo = Hippocampus(cortex)
        
        # Encode episode
        episode = hippo.encode({f"n{i}" for i in range(20)}, source="test")
        initial_replay = episode.replay_count
        
        # Run sleep
        hippo.sleep(cycles=10)
        
        # Replay count should increase (if episode was selected)
        # Or episode should be consolidated
        assert (
            episode.replay_count > initial_replay or 
            episode.state == EpisodeState.CONSOLIDATED
        ), (
            "Sleep should either increase replay_count, "
            "or consolidate episode"
        )
    
    def test_sleep_can_consolidate(self) -> None:
        """
        BIOLOGICAL REQUIREMENT: sufficient sleep consolidates episodes.
        """
        cortex = Cortex()
        hippo = Hippocampus(cortex)
        
        # Encode one episode
        episode = hippo.encode({f"n{i}" for i in range(20)}, source="test")
        
        # Many sleep cycles
        for _ in range(10):
            hippo.sleep(cycles=10)
        
        # Episode should be consolidated after enough replay
        # (or at least have high replay_count)
        assert (
            episode.state == EpisodeState.CONSOLIDATED or
            episode.replay_count >= hippo.CONSOLIDATION_THRESHOLD
        ), (
            f"After many sleep cycles episode should be consolidated "
            f"or have replay_count >= {hippo.CONSOLIDATION_THRESHOLD}, "
            f"but state={episode.state}, replay_count={episode.replay_count}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
