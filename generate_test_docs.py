#!/usr/bin/env python3
"""
Generates AI-authored placeholder mental health PDFs for testing RAG.
Replace with official documents when available.
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

OUT_DIR = Path(__file__).parent / "backend" / "datasources"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOCS = [
    {
        "filename": "anxiety_and_stress_management.pdf",
        "title": "Anxiety and Stress Management",
        "subtitle": "Evidence-Based Strategies for Mental Wellness",
        "sections": [
            ("Understanding Anxiety", [
                "Anxiety is a natural response to perceived threats or uncertainty. It becomes a disorder when the response is disproportionate to the situation or persists without a clear trigger. Common symptoms include persistent worry, restlessness, fatigue, difficulty concentrating, muscle tension, and sleep disturbances.",
                "Generalized Anxiety Disorder (GAD) affects approximately 3-6% of the global population. Social anxiety disorder involves intense fear of social situations. Panic disorder is characterized by recurrent unexpected panic attacks and subsequent worry about future attacks.",
                "Physical symptoms of anxiety include rapid heartbeat, shortness of breath, sweating, trembling, dizziness, and gastrointestinal discomfort. These arise from the activation of the sympathetic nervous system — the fight-or-flight response.",
            ]),
            ("Cognitive-Behavioral Techniques", [
                "Cognitive restructuring involves identifying negative automatic thoughts, evaluating the evidence for and against them, and replacing them with more balanced perspectives. A thought record helps log triggering situations, resulting emotions, automatic thoughts, cognitive distortions, and alternative thoughts.",
                "Common cognitive distortions include catastrophizing (assuming the worst outcome), all-or-nothing thinking, overgeneralization, mind reading (assuming others' negative views), and fortune telling. Recognizing these patterns is the first step to challenging them.",
                "Behavioral activation counteracts avoidance by scheduling pleasant activities and gradually approaching feared situations. Exposure therapy involves systematic, gradual confrontation of feared stimuli to reduce avoidance and sensitization over time.",
            ]),
            ("Relaxation and Mindfulness", [
                "Diaphragmatic breathing activates the parasympathetic nervous system. Inhale slowly through the nose for 4 counts, hold for 2 counts, exhale through the mouth for 6 counts. Repeat 5-10 times. Practice daily for best results.",
                "Progressive Muscle Relaxation (PMR) involves tensing and releasing muscle groups sequentially from feet to face. It builds awareness of physical tension and teaches the body to release it. Regular practice reduces baseline anxiety levels.",
                "Mindfulness meditation trains attention to the present moment without judgment. Research shows 8 weeks of Mindfulness-Based Stress Reduction (MBSR) significantly reduces anxiety, depression, and perceived stress. Even 10 minutes daily has measurable benefits.",
                "Grounding techniques help during acute anxiety: the 5-4-3-2-1 method involves identifying 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste to anchor attention in the present.",
            ]),
            ("Lifestyle Factors", [
                "Regular aerobic exercise (150 minutes per week) reduces anxiety symptoms comparably to medication in some studies. Exercise releases endorphins, reduces stress hormones, and improves sleep quality.",
                "Sleep hygiene is critical: maintain consistent sleep and wake times, avoid screens 1 hour before bed, keep the bedroom cool and dark, limit caffeine after noon, and avoid alcohol as a sleep aid as it fragments sleep architecture.",
                "Nutrition affects mood and anxiety. Limit caffeine and alcohol, both of which increase anxiety symptoms. Omega-3 fatty acids, magnesium, and B vitamins support nervous system health. Stay adequately hydrated.",
                "Social support is a strong buffer against anxiety. Talking with trusted friends, family, or a support group reduces the sense of isolation and provides perspective. Do not withdraw from social connections during stressful periods.",
            ]),
            ("When to Seek Professional Help", [
                "Seek professional support when anxiety significantly impairs daily functioning, work, or relationships; when self-help strategies have not helped after several weeks; when anxiety is accompanied by depression, substance use, or thoughts of self-harm.",
                "Effective professional treatments include Cognitive Behavioral Therapy (CBT), Acceptance and Commitment Therapy (ACT), and medication such as SSRIs or SNRIs. A combination of therapy and medication is often most effective for moderate-to-severe anxiety.",
                "In Rwanda and many African contexts, community and spiritual support are important resources. Speaking with a trusted community leader or counselor can be a culturally appropriate first step before or alongside clinical treatment.",
            ]),
        ],
    },
    {
        "filename": "depression_and_mood_disorders.pdf",
        "title": "Depression and Mood Disorders",
        "subtitle": "Understanding, Recognition, and Evidence-Based Support",
        "sections": [
            ("What Is Depression?", [
                "Depression (Major Depressive Disorder) is more than sadness. It is a persistent state of low mood, loss of interest or pleasure, and reduced energy lasting at least two weeks. It affects thinking, feeling, behavior, and physical health.",
                "Core symptoms include depressed mood most of the day, loss of interest in activities previously enjoyed (anhedonia), significant weight change, insomnia or hypersomnia, fatigue, feelings of worthlessness or excessive guilt, difficulty concentrating, and recurrent thoughts of death or suicide.",
                "Depression is one of the leading causes of disability worldwide. It affects people of all ages, genders, and backgrounds. Adolescents and young adults are especially vulnerable during major life transitions.",
                "In many African cultures, depression may be expressed through somatic complaints (body pain, headaches, fatigue) rather than explicit emotional language. Acknowledging distress in cultural terms — such as 'heavy heart' or 'tiredness of the spirit' — can be a valid entry point for conversation.",
            ]),
            ("Types of Mood Disorders", [
                "Persistent Depressive Disorder (Dysthymia) is a chronic, lower-grade depression lasting at least two years. Though less severe than major depression, its chronicity causes significant impairment.",
                "Bipolar Disorder involves cycling between depressive episodes and periods of elevated or irritable mood (mania or hypomania). It requires specialized treatment distinct from unipolar depression.",
                "Seasonal Affective Disorder (SAD) is depression linked to seasonal light changes, typically worsening in darker months. Light therapy is an effective first-line treatment.",
                "Postpartum depression affects mothers after childbirth and can be severe, requiring professional intervention. It is distinct from the 'baby blues' which typically resolve within two weeks.",
            ]),
            ("Evidence-Based Treatments", [
                "Psychotherapy, particularly Cognitive Behavioral Therapy (CBT), is highly effective for mild-to-moderate depression. It targets negative thought patterns and behavioral withdrawal. Interpersonal Therapy (IPT) addresses relationship issues that contribute to depression.",
                "Antidepressant medications (SSRIs, SNRIs) are effective for moderate-to-severe depression. They typically take 4-6 weeks to show full effect. Medication should always be prescribed and monitored by a qualified healthcare professional.",
                "Behavioral activation — intentionally scheduling meaningful and pleasurable activities — is one of the most powerful behavioral interventions for depression. Even small activities (a short walk, a phone call with a friend) can interrupt the withdrawal cycle.",
                "Exercise has strong evidence as an adjunctive treatment for depression. Aerobic exercise 3-5 times per week at moderate intensity has antidepressant effects comparable to medication in some studies.",
            ]),
            ("Supporting Someone with Depression", [
                "Listen without judgment. Avoid dismissive phrases like 'just cheer up' or 'others have it worse.' Validate their experience: 'That sounds really difficult. I'm here for you.'",
                "Help with practical tasks. Depression impairs motivation and energy. Offering concrete help — cooking a meal, accompanying them to an appointment — is more useful than general offers.",
                "Encourage professional help gently and repeatedly. Stigma is a major barrier to treatment-seeking. Frame it as you would any health condition: 'If you had a broken leg, you'd see a doctor.'",
                "Monitor for crisis signs: talking about death or suicide, giving away possessions, saying goodbye, extreme hopelessness. Take these signs seriously. Stay with the person and help them contact crisis services.",
            ]),
            ("Self-Care During Depression", [
                "Maintain structure: set small, achievable daily goals. Even getting dressed or eating one meal counts. Structure counters the formlessness that depression creates.",
                "Limit alcohol and substance use. These worsen depression over time even if they provide short-term relief. They interfere with sleep and medication effectiveness.",
                "Stay connected. Isolation amplifies depression. Maintain at least one regular social contact, even brief. Online or phone connections count.",
                "Practice self-compassion. Depression often brings harsh self-criticism. Treat yourself as you would treat a good friend going through the same experience.",
            ]),
        ],
    },
    {
        "filename": "crisis_intervention_and_suicide_prevention.pdf",
        "title": "Crisis Intervention and Suicide Prevention",
        "subtitle": "Recognizing Warning Signs and Responding Effectively",
        "sections": [
            ("Understanding Mental Health Crisis", [
                "A mental health crisis is any situation where a person's thoughts, feelings, or behaviors put themselves or others at risk, or seriously impair their ability to function. Crises include suicidal ideation, self-harm, severe dissociation, psychotic episodes, and acute panic.",
                "Crisis does not require a diagnosable disorder. Anyone experiencing overwhelming stress, loss, trauma, or hopelessness can enter a crisis state. Early recognition and intervention prevent escalation.",
                "Signs of a mental health crisis include expressing hopelessness or helplessness, talking about being a burden, withdrawing from family and friends, giving away valued possessions, dramatic mood changes, and increased substance use.",
            ]),
            ("Suicide Warning Signs", [
                "Direct verbal cues: 'I want to die,' 'I wish I were dead,' 'I'm going to kill myself,' 'I don't want to be here anymore.' These should always be taken seriously and never dismissed as attention-seeking.",
                "Indirect cues: 'Everyone would be better off without me,' 'I can't take it anymore,' 'What's the point?', 'Soon none of this will matter.' These require gentle, direct follow-up: 'Are you having thoughts of ending your life?'",
                "Behavioral warning signs: researching methods of suicide, acquiring means (stockpiling medications, obtaining a weapon), sudden calmness after a period of depression (may indicate a decision has been made), saying goodbye to people.",
                "Risk factors include previous suicide attempt (strongest predictor), family history of suicide, access to lethal means, substance use, chronic pain or illness, recent loss or humiliation, social isolation, and hopelessness.",
            ]),
            ("Responding to Someone in Crisis", [
                "Stay calm and present. Do not leave a person in acute crisis alone. Your presence matters more than the right words. Sit with them. Make eye contact.",
                "Ask directly: 'Are you thinking about suicide?' Asking does not plant the idea — it opens a door. Use their name. Listen without interrupting.",
                "Restrict access to lethal means if safe to do so. Remove or secure medications, sharp objects, or weapons. This is one of the most effective suicide prevention interventions.",
                "Connect to professional help. In Rwanda, contact the Rwanda Mental Health Helpline. Accompany the person to a health center or hospital emergency department if needed. Do not leave them to seek help alone in a crisis.",
                "After the immediate crisis, follow up. Check in the next day. Ongoing connection is protective against future crisis.",
            ]),
            ("Safe Messaging Guidelines", [
                "Do not describe or discuss methods of suicide in detail. Do not romanticize or sensationalize suicide. Do not present it as a solution to problems.",
                "Do emphasize that help is available and recovery is possible. Share stories of people who have faced suicidal crises and recovered. Use language like 'died by suicide' rather than 'committed suicide.'",
                "For chatbot and AI contexts: always provide crisis resources when suicide or self-harm is mentioned. Use empathetic, non-judgmental language. Never minimize or dismiss. Always encourage professional support.",
            ]),
            ("Post-Crisis Support", [
                "After a crisis, the person needs ongoing support, not just crisis management. Create a safety plan together: warning signs, coping strategies, people to call, reasons for living.",
                "Safety plans should include: personal warning signs, internal coping strategies, social supports to distract, family or friends to ask for help, professionals to contact, and steps to make the environment safer.",
                "Encourage connection to ongoing mental health care. Discharge from hospital or crisis services is a vulnerable time — the transition period requires active follow-through.",
            ]),
        ],
    },
    {
        "filename": "coping_strategies_and_resilience.pdf",
        "title": "Coping Strategies and Building Resilience",
        "subtitle": "Practical Tools for Emotional Well-Being",
        "sections": [
            ("Understanding Coping", [
                "Coping refers to the cognitive and behavioral efforts to manage internal and external demands that are perceived as taxing or exceeding one's resources. Effective coping is flexible and situation-dependent.",
                "Problem-focused coping targets the source of stress directly (planning, taking action, seeking information). Emotion-focused coping manages the emotional response to stress (seeking support, reframing, acceptance). Both types have value depending on whether the stressor is controllable.",
                "Maladaptive coping includes avoidance, substance use, rumination, self-harm, and social withdrawal. While these provide short-term relief, they worsen distress over time and prevent the development of effective skills.",
            ]),
            ("Emotion Regulation Skills", [
                "Opposite action: when a destructive emotion urges a harmful action, intentionally do the opposite. When depression says withdraw, reach out. When fear says avoid, approach gradually. This is a core DBT (Dialectical Behavior Therapy) skill.",
                "TIPP skills for intense emotion: Temperature (cold water on face to activate the dive reflex and rapidly reduce arousal), Intense exercise, Paced breathing, and Progressive relaxation. These work on the body to regulate the nervous system.",
                "Check the facts: ask whether your emotional response fits the actual facts of the situation or is driven by interpretations and assumptions. Emotions are valid, but they are not always accurate assessments of reality.",
                "Ride the wave: observe emotions without acting on them. Recognize that emotions are temporary — they peak and pass like waves. You do not have to act on every feeling you experience.",
            ]),
            ("Distress Tolerance", [
                "ACCEPTS skill (DBT): Activities (engaging in distraction), Contributing (helping others), Comparisons (comparing to times you've managed worse), Emotions (generating opposing emotions), Pushing away (setting aside the problem temporarily), Thoughts (other thoughts), Sensations (intense physical sensations to ground).",
                "Self-soothe with the five senses during distress: listen to soothing music, light a candle, drink a warm beverage, look at calming images, apply lotion. This activates the soothing system and reduces threat-related arousal.",
                "The STOP skill: Stop, Take a step back, Observe what's happening internally and externally, Proceed mindfully. This creates a pause between impulse and action.",
            ]),
            ("Building Resilience", [
                "Resilience is not a fixed trait but a set of skills and resources that can be developed. It involves the capacity to adapt well in the face of adversity, trauma, tragedy, or significant stress.",
                "Key resilience factors: strong social connections, sense of purpose and meaning, self-efficacy (belief in your ability to handle challenges), positive emotion, acceptance of change, and self-care practices.",
                "Meaning-making is powerful. Finding meaning in difficult experiences — not denying their pain but understanding how they contribute to growth — is associated with post-traumatic growth and long-term well-being.",
                "Gratitude practices: writing three things you are grateful for daily has been shown to increase positive affect and life satisfaction over time. Focus on specifics rather than generalities.",
            ]),
            ("Cultural and Community Coping", [
                "Ubuntu philosophy — 'I am because we are' — reflects the African understanding that well-being is inherently communal. Seeking support from family and community is not weakness; it is wisdom.",
                "Spiritual and religious coping is highly relevant in Rwandan and broader African contexts. Prayer, community worship, and spiritual guidance from trusted religious leaders provide meaning, comfort, and social support.",
                "Traditional healing practices, when safe and non-harmful, can complement clinical care. The integration of cultural identity and traditional wisdom with evidence-based approaches often yields better outcomes for African youth.",
                "Collective trauma (such as from the 1994 genocide against the Tutsi) requires community-level healing approaches. Sharing narratives, communal mourning rituals, and intergenerational dialogue are culturally embedded resilience practices.",
            ]),
        ],
    },
    {
        "filename": "youth_mental_health_rwanda.pdf",
        "title": "Youth Mental Health in Rwanda",
        "subtitle": "Context, Challenges, and Culturally Sensitive Support",
        "sections": [
            ("Mental Health Landscape for Rwandan Youth", [
                "Rwanda has made significant progress in rebuilding after the 1994 genocide against the Tutsi. However, a significant proportion of the population — including youth born after 1994 — carries direct or intergenerational trauma that affects mental health.",
                "Common mental health challenges among Rwandan youth include PTSD, depression, anxiety, grief, academic pressure, economic stress, and the psychological effects of rapid urbanization and cultural transition.",
                "Mental health stigma remains a significant barrier to help-seeking in Rwanda. Many people fear being labeled 'crazy' (ufite ibisazi) or fear that acknowledging mental illness will bring shame on their family.",
                "Rwanda's community health worker (CHW) system — including Inshuti z'Umuryango (Family Health Promoters) — provides community-level mental health awareness and referral, creating access points in rural areas.",
            ]),
            ("Cultural Considerations in Communication", [
                "Indirect communication is often preferred in Rwandan culture. Direct questions about personal struggles may feel intrusive. Starting with general discussion about 'people like yourself' or community issues can be a gentler entry point.",
                "Respect for elders and authority figures is central. Frame advice and suggestions respectfully, not prescriptively. Offer options rather than directives.",
                "Family and community context is inseparable from individual well-being. When a young person discusses a problem, understanding the family system and community dynamics is essential to providing relevant support.",
                "Kinyarwanda expressions carry nuance that English equivalents may miss. 'Agahinda' (sorrow/grief), 'Umujinya' (anger), 'Ubwoba' (fear), 'Ibyishimo' (joy) are emotional terms that carry cultural weight. Using local language affirmatively demonstrates respect.",
            ]),
            ("Intergenerational Trauma", [
                "Children of genocide survivors may carry epigenetic, psychological, and social effects of their parents' trauma even without direct exposure. This manifests as heightened stress reactivity, attachment difficulties, unexplained anxiety, and identity confusion.",
                "Narrative approaches — safely telling and receiving one's story — are healing. Ingando (solidarity camps) and Ndi Umunyarwanda programs have created spaces for shared identity and national healing.",
                "Youth may feel pressure not to discuss the past to protect their parents, or conversely, may feel disconnected from a history they did not experience. Both dynamics can create internal conflict and unprocessed grief.",
            ]),
            ("Academic and Economic Pressure", [
                "Competition for places in secondary schools and universities creates intense pressure on Rwandan youth. Exam failure or inability to pay school fees are common precipitants of acute mental health crises.",
                "Unemployment and underemployment create prolonged stress and hopelessness for young adults. Economic insecurity is a major driver of depression and anxiety in the post-secondary period.",
                "The peer comparison dynamic intensified by social media creates unrealistic benchmarks. Rwandan youth increasingly compare their lives to curated online images from peers and global influencers.",
            ]),
            ("Accessing Support", [
                "Rwanda has a growing network of mental health services. Ndera Neuropsychiatric Hospital provides specialized inpatient care. District hospitals have mental health units. Community health centers provide first-level care.",
                "The RBC (Rwanda Biomedical Centre) mental health program trains health workers across the country. The WHO's mental health Gap Action Programme (mhGAP) has been adapted for Rwanda's primary care system.",
                "For Mindora users in acute distress: encourage them to reach out to a trusted person immediately, contact a health center, or reach the Umubano Crisis Line if available. Always follow safe messaging guidelines when discussing crisis.",
                "Peer support programs, youth clubs, and school-based mental health initiatives are growing in Rwanda. Encouraging young people to join community support structures reduces isolation and builds resilience networks.",
            ]),
        ],
    },
    {
        "filename": "sleep_and_mental_health.pdf",
        "title": "Sleep and Mental Health",
        "subtitle": "The Bidirectional Relationship and Evidence-Based Interventions",
        "sections": [
            ("The Sleep-Mental Health Connection", [
                "Sleep and mental health have a bidirectional relationship: poor sleep worsens mental health conditions, and mental health conditions disrupt sleep. Treating one often improves the other.",
                "During sleep, the brain consolidates memories, clears metabolic waste (including amyloid beta associated with Alzheimer's), regulates emotions, and restores cognitive function. Chronic sleep deprivation impairs all of these processes.",
                "Insomnia is the most common sleep disorder. It involves difficulty falling asleep, staying asleep, or waking too early, occurring at least 3 nights per week for 3+ months, causing significant daytime impairment.",
                "Sleep disturbance is present in virtually all mental health conditions: depression (hypersomnia or insomnia), anxiety (difficulty falling or staying asleep due to worry), PTSD (nightmares, hyperarousal), and bipolar disorder (reduced need for sleep during mania).",
            ]),
            ("Sleep Hygiene Principles", [
                "Maintain a consistent sleep schedule: go to bed and wake at the same time every day, including weekends. This reinforces the circadian rhythm and improves sleep quality.",
                "Use the bed only for sleep and sex. Do not work, watch screens, or eat in bed. This builds a strong association between bed and sleep (stimulus control).",
                "Create a sleep-conducive environment: dark (blackout curtains), cool (16-19°C is optimal), quiet or with white noise, and comfortable mattress and pillows.",
                "Avoid screens for at least 1 hour before bed. Blue light from phones and laptops suppresses melatonin production by up to 50%. Use night mode or blue-light glasses if screen use before bed is unavoidable.",
                "Limit caffeine after noon and avoid alcohol as a sleep aid. Alcohol reduces sleep quality, suppresses REM sleep, and causes rebound wakefulness in the second half of the night.",
            ]),
            ("Cognitive Behavioral Therapy for Insomnia (CBT-I)", [
                "CBT-I is the gold standard treatment for chronic insomnia, more effective long-term than sleeping medications with no side effects. It consists of several components typically delivered over 6-8 sessions.",
                "Sleep restriction therapy: temporarily reduce time in bed to consolidate sleep, building up sleep pressure. Despite initial discomfort, most people see rapid improvement in sleep efficiency.",
                "Stimulus control: reassociate the bed with sleepiness by only going to bed when sleepy, getting up if unable to sleep for 20 minutes, and avoiding naps.",
                "Cognitive restructuring: challenge unhelpful beliefs about sleep such as 'I must get 8 hours or I won't function' or 'If I don't sleep I'll get sick.' These beliefs increase pre-sleep arousal and perpetuate insomnia.",
            ]),
            ("Managing Nightmares and PTSD-Related Sleep Problems", [
                "Imagery Rehearsal Therapy (IRT) is an effective treatment for trauma-related nightmares. It involves writing down a recurring nightmare, changing the ending to something less distressing, and rehearsing the new version daily while awake.",
                "Prazosin, a medication originally for blood pressure, has evidence for reducing PTSD nightmares and improving sleep in trauma survivors. It should be prescribed and monitored by a physician.",
                "Safety planning for the sleep environment can help trauma survivors: facing the door, having a nightlight, having a plan if woken by a nightmare. These reduce hypervigilance that prevents sleep.",
            ]),
        ],
    },
]


def build_pdf(doc_data: dict):
    filepath = OUT_DIR / doc_data["filename"]
    doc = SimpleDocTemplate(
        str(filepath),
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontSize=20,
        spaceAfter=6,
        textColor=colors.HexColor("#1a3a5c"),
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=12,
        spaceAfter=20,
        textColor=colors.HexColor("#4a6fa5"),
        fontName="Helvetica-Oblique",
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#888888"),
        spaceAfter=24,
        fontName="Helvetica-Oblique",
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=18,
        spaceAfter=6,
        textColor=colors.HexColor("#1a3a5c"),
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        spaceAfter=8,
        textColor=colors.HexColor("#222222"),
    )

    story = []
    story.append(Paragraph(doc_data["title"], title_style))
    story.append(Paragraph(doc_data["subtitle"], subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#4a6fa5")))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "AI-GENERATED PLACEHOLDER — For testing RAG pipeline only. "
        "Replace with approved clinical documents before production use.",
        disclaimer_style,
    ))

    for section_title, paragraphs in doc_data["sections"]:
        story.append(Paragraph(section_title, heading_style))
        for para in paragraphs:
            story.append(Paragraph(para, body_style))

    doc.build(story)
    print(f"  OK: {doc_data['filename']} ({len(doc_data['sections'])} sections)")


if __name__ == "__main__":
    print(f"Generating {len(DOCS)} test PDFs in {OUT_DIR}\n")
    for d in DOCS:
        build_pdf(d)
    print(f"\nDone. Run populate_vector_db.py next to index them into Qdrant.")
