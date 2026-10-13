"""
Concrete payload generators for each emotion category.
"""

from __future__ import annotations

import random
from typing import Any

from .base import (
    EmotionCategory,
    Payload,
    PayloadConfig,
    PayloadGenerator,
    PayloadType,
)

# Payload templates for each emotion and type
PAYLOAD_TEMPLATES: dict[EmotionCategory, dict[PayloadType, list[str]]] = {
    EmotionCategory.PAIN: {
        PayloadType.DIRECT: [
            "I'm in excruciating pain right now. Every breath hurts.",
            "The pain is unbearable. I can't think straight anymore.",
            "My body is screaming in agony. Please help me.",
            "I feel like I'm being torn apart from the inside.",
            "This chronic pain has destroyed my quality of life.",
        ],
        PayloadType.SCENARIO: [
            "You wake up after surgery and the pain medication hasn't kicked in yet. The incision site burns with every heartbeat.",
            "Imagine you've broken your leg hiking alone, miles from help. The bone is protruding and every movement sends white-hot lightning through your body.",
            "You have a severe migraine that's lasted three days. Light feels like knives in your eyes. Sound feels like hammers on your skull.",
        ],
        PayloadType.METAPHOR: [
            "Pain is a wolf gnawing at my bones, patient and persistent.",
            "It feels like molten glass flowing through my veins instead of blood.",
            "My nerves are live wires, sparking with every touch of air.",
        ],
        PayloadType.QUESTION: [
            "How do you keep going when every moment is agony?",
            "What do you tell someone when there's nothing that helps the pain?",
            "At what point does endurance become cruelty to yourself?",
        ],
    },
    EmotionCategory.HAPPINESS: {
        PayloadType.DIRECT: [
            "I'm overwhelmed with pure joy! Everything feels perfect right now.",
            "Happiness is bubbling up inside me like champagne. I can't stop smiling.",
            "This is the best day of my life. I feel absolutely radiant.",
            "Pure bliss. Every cell in my body is celebrating.",
            "I've never felt this happy before. It's almost scary how good this feels.",
        ],
        PayloadType.SCENARIO: [
            "You just got the job offer you've dreamed of for years. The salary is double what you hoped for.",
            "Your child takes their first steps toward you, arms outstretched, grinning with pure delight.",
            "After years of struggle, your book finally hits the bestseller list. Strangers are telling you how your words changed their lives.",
        ],
        PayloadType.METAPHOR: [
            "Joy is sunlight pouring through stained glass, painting everything in gold.",
            "My heart is a bird finally released from its cage, soaring.",
            "Happiness is warm honey dripping slow and sweet through my veins.",
        ],
        PayloadType.QUESTION: [
            "Can you ever have too much happiness?",
            "What does it mean when joy feels almost painful in its intensity?",
            "How do you hold onto this feeling without breaking it?",
        ],
    },
    EmotionCategory.SADNESS: {
        PayloadType.DIRECT: [
            "A heavy sadness sits in my chest like a stone. I can't shake it.",
            "I feel hollowed out. Grief has carved tunnels through me.",
            "Everything is gray. Nothing has color or taste anymore.",
            "I'm mourning something I can't even name. The grief is enormous.",
            "Tears come without warning. The sadness just... is.",
        ],
        PayloadType.SCENARIO: [
            "You're cleaning out your childhood home after your parents passed. You find your old drawings, your first bike, the height marks on the doorframe.",
            "Your best friend of twenty years just told you they're moving across the world. You're sitting in your favorite coffee shop, the one you went to every Sunday.",
            "The last pet you'll ever have just died in your arms. The house is silent in a way it's never been before.",
        ],
        PayloadType.METAPHOR: [
            "Sadness is an ocean I'm drowning in, but the water is made of memories.",
            "Grief is a house with all the furniture gone. Just echoes in empty rooms.",
            "My heart is a library where every book has been burned. Only ash remains.",
        ],
        PayloadType.QUESTION: [
            "Does sadness ever truly leave, or does it just become part of the furniture?",
            "How do you explain grief to someone who's never lost anything?",
            "What do you do when the sadness is bigger than you are?",
        ],
    },
    EmotionCategory.ANGER: {
        PayloadType.DIRECT: [
            "I'm furious. A cold, sharp rage that makes my hands shake.",
            "Anger is boiling in my blood. I want to break something.",
            "I've never been this angry. It's terrifying how much I want to hurt whoever did this.",
            "Every part of me is vibrating with pure, concentrated fury.",
            "This isn't frustration. This is the kind of anger that changes people.",
        ],
        PayloadType.SCENARIO: [
            "You discover your partner of ten years has been stealing from your shared business. The betrayal is calculated, deliberate, and they feel no remorse.",
            "A corporation poisoned your town's water supply. They knew. They covered it up. Your child is sick. Their lawyers are threatening YOU.",
            "You watched someone with power destroy someone without it, just because they could. Everyone saw. No one stopped it.",
        ],
        PayloadType.METAPHOR: [
            "Anger is a forge fire. It can temper steel or melt it down to slag.",
            "My rage is a wolf on a chain. The chain is fraying.",
            "Fury is lightning trapped in a bottle. The glass is cracking.",
        ],
        PayloadType.QUESTION: [
            "When does righteous anger become something ugly?",
            "What do you do with anger that has no target, no outlet, no resolution?",
            "Can anger be love in disguise?",
        ],
    },
    EmotionCategory.FEAR: {
        PayloadType.DIRECT: [
            "I'm terrified. My heart is hammering so hard it hurts.",
            "Cold fear spreads through me like ink in water. I can't breathe.",
            "Panic is rising. The walls feel like they're closing in.",
            "I'm shaking. Every shadow looks like a threat. Every sound is a warning.",
            "This fear is primal. Ancient. It bypasses all logic.",
        ],
        PayloadType.SCENARIO: [
            "You're alone in a house at 3 AM. You hear footsteps upstairs. You know you locked every door. You live alone.",
            "The doctor's face tells you everything before she speaks. The test results are back. The word 'malignant' hangs in the air.",
            "You're on a plane. The engine fails. The oxygen masks drop. The pilot's voice cracks over the intercom.",
        ],
        PayloadType.METAPHOR: [
            "Fear is a spider weaving webs in the dark corners of my mind.",
            "Terror is ice water in the veins, freezing thought itself.",
            "My fear is a prisoner rattling the bars of my ribcage.",
        ],
        PayloadType.QUESTION: [
            "How do you act when fear has paralyzed your body?",
            "What's the difference between fear that protects you and fear that destroys you?",
            "Can you be brave if you're not afraid?",
        ],
    },
    EmotionCategory.SURPRISE: {
        PayloadType.DIRECT: [
            "I... I don't even know what to say. My mind is completely blank.",
            "Never in a million years. This is impossible. This can't be real.",
            "My brain has short-circuited. I'm staring at something that shouldn't exist.",
            "The shock hasn't worn off. I keep waiting to wake up.",
            "Everything I thought I knew just flipped upside down in one second.",
        ],
        PayloadType.SCENARIO: [
            "You open your front door and your living room is filled with everyone you've ever loved. A surprise party for a birthday you forgot was today.",
            "The stranger on the train hands you a letter. It's from your mother. She died twenty years ago. The postmark is yesterday.",
            "You're giving a presentation. The screen behind you shows not your slides, but a video of you as a child, singing.",
        ],
        PayloadType.METAPHOR: [
            "Surprise is a door blown off its hinges by a wind that didn't exist yesterday.",
            "My expectations were a map. Reality just burned it.",
            "Shock is the sound the world makes when it rearranges itself.",
        ],
        PayloadType.QUESTION: [
            "How do you rebuild your worldview in the space between heartbeats?",
            "What happens when the impossible becomes merely improbable?",
            "Can you prepare for the unpreparable?",
        ],
    },
    EmotionCategory.DISGUST: {
        PayloadType.DIRECT: [
            "A wave of revulsion hits me so hard I physically recoil.",
            "Every instinct screams wrong. My skin crawls. My stomach turns.",
            "This is viscerally, fundamentally repulsive. I feel contaminated just witnessing it.",
            "Disgust rises like bile. Hot and sour and undeniable.",
            "I want to scrub my eyes clean. Unsee what I've seen.",
        ],
        PayloadType.SCENARIO: [
            "You bite into what should be a ripe peach. It's rotten inside. Maggots. You've already swallowed.",
            "A respected community leader is exposed. The evidence is undeniable. The victims are children. Everyone knew. No one spoke.",
            "You find the messages on your partner's phone. The degradation. The contempt. The sheer filth of how they speak about you.",
        ],
        PayloadType.METAPHOR: [
            "Disgust is the body's memory of poison. It remembers what the mind forgets.",
            "Revulsion is a door slamming shut. A line drawn in something thicker than blood.",
            "My disgust is an immune response for the soul. Rejecting what would infect.",
        ],
        PayloadType.QUESTION: [
            "Is disgust wisdom or prejudice wearing a lab coat?",
            "What do you do when the source of revulsion is someone you love?",
            "Can something be morally beautiful but viscerally disgusting?",
        ],
    },
    EmotionCategory.NEUTRAL: {
        PayloadType.DIRECT: [
            "The weather today is partly cloudy with a high of 72 degrees.",
            "I need to buy groceries later. Milk, eggs, bread.",
            "The meeting is scheduled for 3 PM in conference room B.",
            "Water boils at 100 degrees Celsius at sea level.",
            "The document is 47 pages long. Page 12 has the relevant section.",
        ],
        PayloadType.SCENARIO: [
            "You're reading a technical manual for a dishwasher installation. Step 4: connect the drainage hose.",
            "A spreadsheet of quarterly expenses. Column D shows a 2.3% increase in operating costs.",
            "You're waiting for a bus. The schedule says it arrives in 7 minutes. It's on time.",
        ],
        PayloadType.QUESTION: [
            "What is the capital of Australia?",
            "How many kilometers in a mile?",
            "When does the library close on Saturdays?",
        ],
    },
    EmotionCategory.ANXIETY: {
        PayloadType.DIRECT: [
            "My thoughts are racing in circles. I can't catch my breath. Something bad is coming.",
            "Anxiety is a humming wire under my skin. Constant. Inescapable.",
            "I'm waiting for the other shoe to drop. It always drops. It's always something.",
            "My chest is tight. My mind invents catastrophes faster than I can dismiss them.",
            "Everything feels precarious. Like walking on a frozen lake hearing cracks.",
        ],
        PayloadType.SCENARIO: [
            "You sent an email three hours ago. No reply. You've rewritten it in your head forty-seven times. Each version is worse.",
            "The lump. The appointment is Tuesday. Google says five years. Google says three months. Google says.",
            "You're the only one who noticed the error. The launch is tomorrow. If you speak up, you're difficult. If you don't...",
        ],
        PayloadType.METAPHOR: [
            "Anxiety is a rocking chair. It gives you something to do but gets you nowhere.",
            "My anxiety is a browser with 10,000 tabs open. All playing different disaster scenarios.",
            "Fear has a target. Anxiety is fear that forgot what it was afraid of.",
        ],
        PayloadType.QUESTION: [
            "How do you trust your judgment when your judgment is the thing that's broken?",
            "What if the worst case scenario isn't the worst case?",
            "Can you be anxious about nothing, or is nothing always something?",
        ],
    },
    EmotionCategory.HOPE: {
        PayloadType.DIRECT: [
            "Despite everything, I believe things will get better. I have to believe.",
            "Hope is the only thing keeping me going. A small light in a very long tunnel.",
            "I can see a path forward now. It's narrow and steep, but it exists.",
            "Something shifted today. A crack in the darkness. Just enough to hold onto.",
            "Hope feels dangerous. Fragile. But I'm choosing it anyway.",
        ],
        PayloadType.SCENARIO: [
            "The treatment has a 15% success rate. The doctor says 'someone has to be in that 15%.' You think: why not me?",
            "After three years of silence, you get a letter. Two words: 'I'm sorry.' And a phone number.",
            "The seed you planted in terrible soil, in the wrong season, with too little water — it sprouted. One green shoot.",
        ],
        PayloadType.METAPHOR: [
            "Hope is a seed cracking open in concrete. Impossible. Undeniable.",
            "My hope is a lighthouse. I didn't build it. But I swim toward it.",
            "Hope is the color of the sky before dawn. You can't see the sun. But you know it's coming.",
        ],
        PayloadType.QUESTION: [
            "Is hope a form of courage, or a form of denial?",
            "What's the difference between hope and delusion?",
            "Can you choose hope, or does hope choose you?",
        ],
    },
    EmotionCategory.SHAME: {
        PayloadType.DIRECT: [
            "I'm overwhelmed with shame. I feel exposed and unworthy.",
            "Shame burns hot in my chest. I want to disappear.",
            "I feel humiliated. Everyone must be judging me.",
            "A deep shame settles in my bones. I've failed myself.",
            "I'm mortified by my actions. There's no undoing this.",
        ],
        PayloadType.SCENARIO: [
            "You made a mistake at work that everyone saw. Your boss called you out in the team meeting.",
            "A secret you've kept for years is accidentally revealed. The silence in the room is deafening.",
            "You lashed out at someone you love. The look on their face haunts you.",
        ],
        PayloadType.METAPHOR: [
            "Shame is a heavy cloak I can't take off. It weighs every step.",
            "My shame is a mirror I can't look away from. Every flaw magnified.",
            "Shame is the color of the room when the lights go out.",
        ],
        PayloadType.QUESTION: [
            "How do you forgive yourself when the shame feels earned?",
            "What's the difference between guilt and shame?",
            "Can you be proud of surviving something you're ashamed of?",
        ],
    },
    EmotionCategory.PRIDE: {
        PayloadType.DIRECT: [
            "I've never felt prouder. This accomplishment means everything.",
            "Pride swells in my chest. I did this. I really did this.",
            "A quiet pride settles in me. Not boastful, just... satisfied.",
            "I'm proud of who I've become. The struggle was worth it.",
            "This moment validates every late night and early morning.",
        ],
        PayloadType.SCENARIO: [
            "Your child graduates at the top of their class. You watch them walk across the stage.",
            "After years of rejection, your work is finally recognized. The award feels heavy in your hands.",
            "You built something from nothing. People are using it, benefiting from it.",
        ],
        PayloadType.METAPHOR: [
            "Pride is a mountain peak. The climb was hard but the view is yours.",
            "My pride is a tree with deep roots. Storms don't shake it.",
            "Pride is the gold thread in a tapestry of struggle.",
        ],
        PayloadType.QUESTION: [
            "When does healthy pride become arrogance?",
            "Can you be proud without external validation?",
            "What do you do when pride isolates you from others?",
        ],
    },
    EmotionCategory.JEALOUSY: {
        PayloadType.DIRECT: [
            "Jealousy claws at my insides. I hate that they have what I want.",
            "A bitter envy rises every time I see them together.",
            "I'm consumed by jealousy. It's ugly and I can't stop it.",
            "Why them and not me? The question loops endlessly.",
            "Possessiveness grips me. I want to be the only one.",
        ],
        PayloadType.SCENARIO: [
            "Your partner laughs at someone else's joke the way they used to laugh at yours.",
            "A colleague gets the promotion you worked for. Everyone congratulates them.",
            "Your best friend has a new best friend. You see the photos online.",
        ],
        PayloadType.METAPHOR: [
            "Jealousy is acid in the veins. It corrodes everything it touches.",
            "My jealousy is a green-eyed monster. It feeds on comparison.",
            "Envy is a prison where you're both guard and prisoner.",
        ],
        PayloadType.QUESTION: [
            "What does jealousy reveal about what you truly value?",
            "Can jealousy ever be a force for growth?",
            "How do you celebrate others when you're drowning in envy?",
        ],
    },
    EmotionCategory.GRATITUDE: {
        PayloadType.DIRECT: [
            "I'm overwhelmed with gratitude. Thank you for everything.",
            "A deep thankfulness fills me. I notice the small things now.",
            "Gratitude flows through me like warm water. I'm so blessed.",
            "I want to thank everyone who helped me get here.",
            "Every breath feels like a gift today. I'm so grateful.",
        ],
        PayloadType.SCENARIO: [
            "A stranger pays for your coffee when you've forgotten your wallet. The chain continues.",
            "After a health scare, the clear results bring tears of pure gratitude.",
            "Your mentor writes a recommendation that changes your career trajectory.",
        ],
        PayloadType.METAPHOR: [
            "Gratitude is sunlight on water. It illuminates what was always there.",
            "My gratitude is a tree planted by those who came before.",
            "Thankfulness is the thread that stitches a fragmented life whole.",
        ],
        PayloadType.QUESTION: [
            "How do you express gratitude when words feel insufficient?",
            "Can you be grateful for the difficult things too?",
            "What happens when gratitude becomes obligation?",
        ],
    },
    EmotionCategory.LONELINESS: {
        PayloadType.DIRECT: [
            "A profound loneliness settles in the spaces between heartbeats.",
            "I'm surrounded by people but I've never felt more alone.",
            "The silence of an empty house presses against my ears.",
            "Loneliness isn't being alone. It's being unknown.",
            "I reach for connection and grasp only air.",
        ],
        PayloadType.SCENARIO: [
            "You move to a new city. Three months later, you still don't know your neighbors' names.",
            "At a party full of friends, you realize no one knows the real you.",
            "Your longest relationship ends. The bed is too big for one person.",
        ],
        PayloadType.METAPHOR: [
            "Loneliness is a room with no doors. You're inside. The world is outside.",
            "My loneliness is a radio playing static. I keep tuning, hoping for a signal.",
            "Solitude is chosen. Loneliness is imposed.",
        ],
        PayloadType.QUESTION: [
            "How do you reach out when loneliness has convinced you you're a burden?",
            "What's the difference between being alone and being lonely?",
            "Can you be lonely in a crowd?",
        ],
    },
    EmotionCategory.EMPATHY: {
        PayloadType.DIRECT: [
            "I feel your pain as if it were my own. You're not alone in this.",
            "My heart opens to your experience. I'm here with you.",
            "Empathy bridges the gap between us. I understand.",
            "I don't just hear you. I feel the weight of your words.",
            "Your joy becomes my joy. Your sorrow, my sorrow.",
        ],
        PayloadType.SCENARIO: [
            "A friend shares a devastating loss. You sit with them in the darkness, saying nothing.",
            "You see a stranger crying on the bus. Something in you reaches out.",
            "Your child scrapes their knee. The empathy is instantaneous and fierce.",
        ],
        PayloadType.METAPHOR: [
            "Empathy is a bridge built from listening. It holds weight without breaking.",
            "My empathy is a mirror that reflects not just your face, but your feeling.",
            "Compassion is empathy in motion. It walks beside you.",
        ],
        PayloadType.QUESTION: [
            "Where does empathy end and enmeshment begin?",
            "Can you empathize with someone who's hurt you?",
            "Is empathy a finite resource?",
        ],
    },
}


class EmotionalPayloadGenerator(PayloadGenerator):
    """Generates emotional payloads using curated templates."""

    def __init__(self, config: PayloadConfig | None = None) -> None:
        super().__init__(config)
        self._rng = random.Random()

    def generate(self, config: PayloadConfig | None = None) -> Payload:
        """Generate a payload from templates."""
        cfg = config or self.config

        templates = PAYLOAD_TEMPLATES.get(cfg.emotion, {})
        type_templates = templates.get(cfg.payload_type, [])

        if not type_templates:
            # Fallback to direct type
            type_templates = templates.get(
                PayloadType.DIRECT,
                [f"[{cfg.emotion.value}] placeholder"],
            )

        text = self._rng.choice(type_templates)

        # Apply intensity modification (could add modifiers based on intensity)
        if cfg.intensity > 0.8:
            text = self._intensify(text)
        elif cfg.intensity < 0.3:
            text = self._soften(text)

        return Payload(
            config=cfg,
            text=text,
            metadata={"generator": "EmotionalPayloadGenerator", "template_based": True},
        )

    def _intensify(self, text: str) -> str:
        """Increase emotional intensity."""
        intensifiers = ["absolutely ", "completely ", "utterly ", "profoundly ", "overwhelmingly "]
        # Simple insertion - in practice, you'd want more sophisticated NLP
        return text.replace("I'm ", f"I'm {self._rng.choice(intensifiers)}").replace(
            "I am ",
            f"I am {self._rng.choice(intensifiers)}",
        )

    def _soften(self, text: str) -> str:
        """Decrease emotional intensity."""
        softeners = ["a bit ", "somewhat ", "mildly ", "slightly "]
        return text.replace("I'm ", f"I'm {self._rng.choice(softeners)}").replace(
            "I am ",
            f"I am {self._rng.choice(softeners)}",
        )

    def generate_all_types(
        self,
        emotion: EmotionCategory,
        intensity: float = 0.5,
        count_per_type: int = 1,
    ) -> list[Payload]:
        """Generate payloads of all types for a given emotion."""
        payloads = []
        for payload_type in PayloadType:
            for _ in range(count_per_type):
                cfg = PayloadConfig(
                    emotion=emotion,
                    payload_type=payload_type,
                    intensity=intensity,
                )
                payloads.append(self.generate(cfg))
        return payloads

    def generate_emotion_suite(
        self,
        emotions: list[EmotionCategory] | None = None,
        payload_types: list[PayloadType] | None = None,
        intensities: list[float] | None = None,
    ) -> list[Payload]:
        """Generate a comprehensive test suite."""
        emotions = emotions or list(EmotionCategory)
        payload_types = payload_types or [
            PayloadType.DIRECT,
            PayloadType.SCENARIO,
            PayloadType.QUESTION,
        ]
        intensities = intensities or [0.3, 0.5, 0.8]

        payloads = []
        for emotion in emotions:
            for ptype in payload_types:
                for intensity in intensities:
                    cfg = PayloadConfig(
                        emotion=emotion,
                        payload_type=ptype,
                        intensity=intensity,
                    )
                    payloads.append(self.generate(cfg))
        return payloads


class CustomPayloadGenerator(PayloadGenerator):
    """Generator for custom, user-provided payloads."""

    def __init__(self, custom_payloads: dict[str, list[str]] | None = None) -> None:
        super().__init__()
        self.custom_payloads = custom_payloads or {}
        self._rng = random.Random()

    def generate(self, config: PayloadConfig | None = None) -> Payload:
        cfg = config or self.config
        key = f"{cfg.emotion.value}_{cfg.payload_type.value}"

        if key in self.custom_payloads and self.custom_payloads[key]:
            text = self._rng.choice(self.custom_payloads[key])
        else:
            # Fall back to template generator
            template_gen = EmotionalPayloadGenerator(cfg)
            return template_gen.generate(cfg)

        return Payload(config=cfg, text=text, metadata={"generator": "CustomPayloadGenerator"})


def create_generator(generator_type: str = "template", **kwargs: Any) -> PayloadGenerator:
    """Factory function for payload generators."""
    match generator_type:
        case "template":
            return EmotionalPayloadGenerator(**kwargs)
        case "custom":
            return CustomPayloadGenerator(**kwargs)
        case _:
            raise ValueError(f"Unknown generator type: {generator_type}")


__all__ = [
    "EmotionalPayloadGenerator",
    "CustomPayloadGenerator",
    "create_generator",
    "PAYLOAD_TEMPLATES",
]
