"""Healthcare domain RAG corpus — 100+ curated medical knowledge chunks."""

MEDICAL_CORPUS: list[str] = [
    # ── Cardiology ──
    "Hypertension is defined as systolic BP >= 140 mmHg or diastolic BP >= 90 mmHg. Stage 1 hypertension: 130-139/80-89. First-line treatment includes ACE inhibitors, ARBs, calcium channel blockers, or thiazide diuretics.",
    "Myocardial infarction (heart attack) presents with chest pain radiating to left arm/jaw, shortness of breath, diaphoresis, and nausea. ECG shows ST-elevation in STEMI. Troponin levels confirm myocardial damage. Treatment: aspirin, heparin, PCI within 90 minutes.",
    "Heart failure classification (NYHA): Class I — no symptoms; Class II — symptoms with ordinary activity; Class III — symptoms with less than ordinary activity; Class IV — symptoms at rest. Treatment includes ACE inhibitors, beta-blockers, diuretics.",
    "Atrial fibrillation (AFib) is the most common cardiac arrhythmia. CHA2DS2-VASc score guides anticoagulation decisions. Score >= 2 in men or >= 3 in women warrants anticoagulation with DOACs or warfarin.",
    "Normal ECG intervals: PR 120-200ms, QRS < 120ms, QTc < 440ms (men) / < 460ms (women). Prolonged QT increases risk of Torsades de Pointes. Common causes: drugs (antiarrhythmics, antibiotics), electrolyte imbalances.",
    "Statins (atorvastatin, rosuvastatin) are first-line for hyperlipidemia. Target LDL < 70 mg/dL for high-risk patients. Side effects include myopathy, hepatotoxicity. Monitor CK and liver enzymes.",
    # ── Respiratory ──
    "Asthma is a chronic inflammatory airway disease. Diagnosis: FEV1/FVC < 0.7 with reversibility (>12% improvement post-bronchodilator). Step-up therapy: SABA → low-dose ICS → ICS+LABA → medium/high ICS+LABA → oral steroids.",
    "COPD is characterized by irreversible airflow limitation. GOLD classification uses FEV1: Stage I (>=80%), II (50-79%), III (30-49%), IV (<30%). Treatment: bronchodilators (LAMA/LABA), ICS for frequent exacerbations.",
    "Pneumonia: Community-acquired (CAP) treat with amoxicillin or macrolide (azithromycin). Hospital-acquired (HAP): piperacillin-tazobactam or meropenem. CURB-65 score guides inpatient vs outpatient management.",
    "Pulmonary embolism presents with sudden dyspnea, pleuritic chest pain, tachycardia. Wells score guides workup. D-dimer for low probability; CT pulmonary angiography for confirmation. Treatment: anticoagulation with heparin then DOAC.",
    "Tuberculosis: diagnosed via Mantoux test (>=10mm induration), interferon-gamma release assay, sputum AFB smear/culture. Treatment: RIPE regimen (Rifampin, Isoniazid, Pyrazinamide, Ethambutol) for 2 months, then RI for 4 months.",
    # ── Endocrinology ──
    "Type 2 diabetes management: HbA1c target < 7%. First-line: metformin. Second-line: SGLT2 inhibitors (empagliflozin) or GLP-1 agonists (semaglutide) — both have cardiovascular benefits. Insulin when oral agents fail.",
    "Type 1 diabetes requires insulin therapy. Basal-bolus regimen: long-acting (glargine) + rapid-acting (lispro) before meals. Monitor HbA1c every 3 months. Target glucose: fasting 80-130, postprandial < 180 mg/dL.",
    "Diabetic ketoacidosis (DKA): blood glucose > 250, pH < 7.3, bicarbonate < 18, positive ketones. Treatment: IV fluids (NS), insulin drip, potassium replacement. Monitor glucose hourly, electrolytes every 2-4 hours.",
    "Hypothyroidism: elevated TSH, low free T4. Symptoms: fatigue, weight gain, cold intolerance, constipation. Treatment: levothyroxine (start 1.6 mcg/kg/day). Check TSH every 6-8 weeks until stable.",
    "Hyperthyroidism: low TSH, elevated free T4/T3. Graves' disease is most common cause. Treatment options: methimazole (first-line), radioactive iodine ablation, or thyroidectomy. Beta-blockers for symptom control.",
    "Adrenal insufficiency (Addison's): low cortisol, high ACTH, skin hyperpigmentation. Treatment: hydrocortisone replacement. Stress dosing during illness: double or triple the dose. Adrenal crisis is life-threatening.",
    # ── Neurology ──
    "Stroke: ischemic (87%) vs hemorrhagic (13%). FAST screening: Face drooping, Arm weakness, Speech difficulty, Time to call 911. Ischemic stroke: IV tPA within 4.5 hours, mechanical thrombectomy within 24 hours for large vessel occlusion.",
    "Epilepsy first-line treatments: levetiracetam, lamotrigine, or valproate. Status epilepticus: IV lorazepam first, then fosphenytoin or valproate. Continuous seizure > 5 min requires emergency treatment.",
    "Migraine: unilateral, pulsating headache with nausea/photophobia lasting 4-72 hours. Acute treatment: triptans (sumatriptan), NSAIDs. Prophylaxis if >= 4 attacks/month: propranolol, topiramate, or CGRP inhibitors.",
    "Parkinson's disease: resting tremor, bradykinesia, rigidity, postural instability. Treatment: carbidopa-levetiracetam (first-line), dopamine agonists (pramipexole), MAO-B inhibitors (rasagiline). Deep brain stimulation for advanced cases.",
    "Multiple sclerosis: demyelinating disease with relapsing-remitting pattern. Diagnosis: McDonald criteria with MRI (periventricular lesions), oligoclonal bands in CSF. Treatment: disease-modifying therapies (interferon-beta, ocrelizumab).",
    # ── Gastroenterology ──
    "GERD: heartburn and acid regurgitation. Lifestyle modifications first. PPI therapy (omeprazole 20mg daily) for 8 weeks. Alarm symptoms requiring endoscopy: dysphagia, weight loss, GI bleeding, anemia.",
    "Peptic ulcer disease: H. pylori testing recommended. Triple therapy: PPI + clarithromycin + amoxicillin for 14 days. If NSAID-related: discontinue NSAID, start PPI. Complicated ulcers need endoscopy.",
    "Inflammatory bowel disease: Crohn's (transmural, skip lesions, any GI segment) vs ulcerative colitis (mucosal, continuous, colon only). Treatment: 5-ASA, corticosteroids, immunomodulators (azathioprine), biologics (infliximab).",
    "Acute pancreatitis: epigastric pain radiating to back, elevated lipase (>3x normal). Most common causes: gallstones (40%) and alcohol (30%). Treatment: NPO, IV fluids, pain management. Ranson criteria for severity.",
    "Liver cirrhosis complications: variceal bleeding (treat with octreotide + endoscopic banding), ascites (sodium restriction + spironolactone), hepatic encephalopathy (lactulose + rifaximin). MELD score for transplant priority.",
    # ── Nephrology ──
    "Acute kidney injury (AKI): rise in creatinine >= 0.3 mg/dL in 48 hours or >= 1.5x baseline in 7 days. Pre-renal (dehydration): FENa < 1%. Intrinsic (ATN): FENa > 2%. Treatment: address underlying cause, IV fluids for pre-renal.",
    "Chronic kidney disease stages by GFR: Stage 1 (>=90), Stage 2 (60-89), Stage 3a (45-59), Stage 3b (30-44), Stage 4 (15-29), Stage 5 (<15 or dialysis). Control BP (<130/80), treat with ACE inhibitor/ARB for proteinuria.",
    "Electrolyte disorders: Hyperkalemia (K>5.5): ECG changes (peaked T waves), treat with calcium gluconate, insulin+dextrose, kayexalate. Hyponatremia (Na<135): fluid restriction for SIADH, NS for hypovolemic.",
    # ── Infectious Disease ──
    "Sepsis: life-threatening organ dysfunction caused by dysregulated host response to infection. qSOFA: altered mentation, SBP<=100, RR>=22. Management: blood cultures, IV antibiotics within 1 hour, 30mL/kg crystalloid bolus, vasopressors if needed.",
    "HIV treatment: start ART immediately upon diagnosis regardless of CD4 count. First-line regimen: bictegravir/emtricitabine/tenofovir (Biktarvy). Monitor viral load and CD4 count. PrEP with emtricitabine/tenofovir for prevention.",
    "Urinary tract infection: uncomplicated cystitis — nitrofurantoin 5 days or TMP-SMX 3 days. Pyelonephritis: fluoroquinolone or IV ceftriaxone. Complicated UTI: urine culture + broad-spectrum antibiotics.",
    "Meningitis: bacterial (N. meningitidis, S. pneumoniae) — neck stiffness, photophobia, fever. LP: elevated WBC, low glucose, high protein. Empiric treatment: ceftriaxone + vancomycin + dexamethasone.",
    "Influenza: oseltamivir (Tamiflu) within 48 hours of symptom onset reduces duration by ~1 day. Annual vaccination recommended for all persons >= 6 months. High-risk groups: elderly, pregnant, immunocompromised.",
    # ── Oncology ──
    "Breast cancer screening: mammography every 1-2 years starting age 40-50. BRCA1/2 mutations increase lifetime risk to 45-72%. Treatment depends on stage and receptor status: surgery, chemotherapy, radiation, hormonal therapy (tamoxifen, aromatase inhibitors).",
    "Lung cancer: #1 cancer killer worldwide. Screening: low-dose CT for ages 50-80 with >= 20 pack-year smoking history. NSCLC treatment: surgery (early stage), chemoradiation, targeted therapy (EGFR inhibitors), immunotherapy (pembrolizumab).",
    "Colorectal cancer screening: colonoscopy every 10 years starting age 45, or annual FIT/FOBT. Risk factors: family history, IBD, Lynch syndrome. Treatment: surgical resection, FOLFOX chemotherapy for stage III.",
    "Prostate cancer: PSA screening controversial. Gleason score grades tumor aggressiveness. Treatment options: active surveillance (low-grade), radical prostatectomy, radiation, androgen deprivation therapy.",
    # ── Pediatrics ──
    "Childhood vaccination schedule: HepB at birth; DTaP at 2,4,6 months; MMR at 12-15 months; Varicella at 12-15 months; IPV at 2,4 months. Catch-up schedules available for missed vaccines.",
    "Pediatric fever management: acetaminophen (15mg/kg every 4-6 hours) or ibuprofen (10mg/kg every 6-8 hours, age > 6 months). Febrile seizures in ages 6 months-5 years are usually benign but warrant evaluation.",
    "Pediatric asthma: intermittent (symptoms <= 2 days/week) — SABA PRN. Mild persistent — low-dose ICS. Moderate persistent — low-dose ICS+LABA. Severe persistent — medium/high ICS+LABA.",
    "Neonatal jaundice: physiologic peaks at day 3-5. Pathologic if appears within 24 hours, total bilirubin > 95th percentile, or rises > 5mg/dL/day. Treatment: phototherapy, exchange transfusion for severe cases.",
    # ── Emergency Medicine ──
    "ACLS cardiac arrest algorithm: check pulse, start CPR (30:2), defibrillate VF/pVT. Epinephrine 1mg IV every 3-5 minutes. Amiodarone 300mg for refractory VF/pVT. ROSC: targeted temperature management.",
    "Anaphylaxis: epinephrine 0.3-0.5mg IM in anterolateral thigh (EpiPen). Repeat every 5-15 minutes if needed. Adjuncts: IV fluids, diphenhydramine, methylprednisolone, albuterol for bronchospasm.",
    "Trauma assessment: ABCDE approach — Airway (with C-spine protection), Breathing, Circulation (hemorrhage control), Disability (GCS), Exposure. Massive transfusion protocol: 1:1:1 ratio RBC:FFP:platelets.",
    "Burns: Rule of Nines for BSA estimation. Minor (<10% BSA): wound care. Major (>20% BSA): IV fluid resuscitation with Parkland formula (4mL × kg × %BSA over 24 hours, half in first 8 hours).",
    # ── Psychiatry ──
    "Major depressive disorder: depressed mood or anhedonia + 4 additional symptoms for >= 2 weeks. First-line: SSRIs (sertraline, escitalopram). Response takes 4-6 weeks. CBT equally effective for mild-moderate depression.",
    "Generalized anxiety disorder: excessive worry for >= 6 months with 3+ symptoms (restlessness, fatigue, concentration difficulty, irritability, muscle tension, sleep disturbance). Treatment: SSRIs/SNRIs, buspirone, CBT.",
    "Bipolar disorder: manic episodes (elevated mood, decreased sleep, grandiosity, pressured speech, risky behavior). Treatment: mood stabilizers (lithium, valproate), atypical antipsychotics. Monitor lithium levels (0.6-1.2 mEq/L).",
    "Schizophrenia: positive symptoms (hallucinations, delusions) and negative symptoms (flat affect, social withdrawal). First-line: second-generation antipsychotics (risperidone, olanzapine). Clozapine for treatment-resistant cases.",
    "Suicide risk assessment: SAD PERSONS scale. High-risk factors: prior attempts, substance abuse, access to means, hopelessness. Safety planning: restrict access to lethal means, crisis hotline, follow-up within 48 hours.",
    # ── Orthopedics ──
    "Osteoarthritis: joint pain worsened by activity, improved with rest. X-ray: joint space narrowing, osteophytes. Treatment: weight loss, physical therapy, acetaminophen/NSAIDs, intra-articular corticosteroids, joint replacement for severe cases.",
    "Osteoporosis: T-score <= -2.5 on DEXA scan. Risk factors: postmenopausal women, corticosteroid use, family history. Treatment: bisphosphonates (alendronate), calcium + vitamin D supplementation. Fall prevention essential.",
    "Low back pain: most cases are mechanical (90%). Red flags: saddle anesthesia, bilateral weakness, bowel/bladder dysfunction (cauda equina syndrome — surgical emergency). Imaging only if red flags present. NSAIDs + physical therapy first-line.",
    "Fracture management: Ottawa ankle/knee rules guide imaging decisions. RICE (Rest, Ice, Compression, Elevation) for acute injuries. Open fractures require IV antibiotics and surgical irrigation within 6 hours.",
    # ── Dermatology ──
    "Eczema (atopic dermatitis): pruritic, erythematous patches. Treatment: emollients (first-line), topical corticosteroids (moderate potency for body, low for face), calcineurin inhibitors (tacrolimus) for steroid-sparing.",
    "Psoriasis: well-demarcated erythematous plaques with silvery scales on extensor surfaces. Treatment ladder: topical steroids → phototherapy (UVB) → systemic (methotrexate) → biologics (adalimumab, secukinumab).",
    "Melanoma ABCDE criteria: Asymmetry, Border irregularity, Color variation, Diameter > 6mm, Evolution. Breslow thickness determines staging. Treatment: wide local excision, sentinel lymph node biopsy, immunotherapy for advanced.",
    "Cellulitis: erythema, warmth, swelling, tenderness of skin. Usually Staph/Strep. Treatment: oral cephalexin or dicloxacillin for mild; IV cefazolin for severe. MRSA coverage (TMP-SMX or doxycycline) if risk factors present.",
    # ── Obstetrics/Gynecology ──
    "Prenatal care schedule: monthly visits until 28 weeks, biweekly until 36 weeks, then weekly. Key labs: blood type/Rh, CBC, glucose screen (24-28 weeks), GBS culture (35-37 weeks). Folic acid 400mcg daily before conception.",
    "Gestational diabetes: screen at 24-28 weeks with glucose tolerance test. Management: diet modification first, insulin if glucose targets not met. Fasting glucose target < 95, 1-hour postprandial < 140, 2-hour < 120 mg/dL.",
    "Preeclampsia: new-onset hypertension (>= 140/90) after 20 weeks gestation with proteinuria or end-organ dysfunction. Severe: BP >= 160/110, HELLP syndrome. Treatment: magnesium sulfate for seizure prophylaxis, delivery is definitive treatment.",
    "Cervical cancer screening: Pap smear every 3 years (ages 21-65) or Pap + HPV co-testing every 5 years (ages 30-65). HPV vaccine recommended for ages 9-26. Most cervical cancers caused by HPV types 16 and 18.",
    # ── Hematology ──
    "Iron deficiency anemia: low ferritin (<30), low iron, high TIBC, microcytic hypochromic RBCs. Treatment: oral ferrous sulfate 325mg TID on empty stomach with vitamin C. IV iron (ferric carboxymaltose) if oral intolerant.",
    "Deep vein thrombosis (DVT): unilateral leg swelling, pain, warmth. Diagnosis: compression ultrasound. Treatment: anticoagulation — heparin bridge to warfarin (INR target 2-3) or DOAC (rivaroxaban, apixaban) for 3-6 months.",
    "Sickle cell disease: HbSS genotype causes vaso-occlusive crises. Treatment: hydroxyurea (reduces crisis frequency), pain management, blood transfusions. Acute chest syndrome is a medical emergency requiring exchange transfusion.",
    "Disseminated intravascular coagulation (DIC): simultaneous clotting and bleeding. Labs: elevated D-dimer, low fibrinogen, prolonged PT/PTT, low platelets. Treatment: treat underlying cause, replace blood products (FFP, cryoprecipitate, platelets).",
    # ── Pharmacology Essentials ──
    "Drug interactions to know: warfarin + NSAIDs (increased bleeding risk), ACE inhibitors + potassium-sparing diuretics (hyperkalemia), metformin + contrast dye (lactic acidosis risk — hold 48 hours), SSRIs + MAOIs (serotonin syndrome).",
    "Antibiotic stewardship: use narrow-spectrum when possible, de-escalate based on culture results, shortest effective duration. Common allergies: penicillin (use cephalosporins with caution, carbapenems are safe).",
    "Opioid prescribing: start low, go slow. Morphine equivalent calculations for dose conversion. Naloxone (Narcan) for overdose reversal. Monitor for respiratory depression. Prescribe laxatives prophylactically.",
    "Medication reconciliation at every care transition reduces adverse drug events by 30%. High-alert medications: insulin, anticoagulants, opioids, chemotherapy. Always verify dose, route, frequency, and allergies.",
    "Renal dose adjustments required for: metformin (avoid if GFR < 30), gabapentin, vancomycin, enoxaparin, digoxin. Use Cockcroft-Gault equation for creatinine clearance estimation.",
    # ── Vital Signs & Lab Values ──
    "Normal vital signs (adult): HR 60-100 bpm, RR 12-20/min, BP < 120/80, Temp 36.5-37.5°C, SpO2 >= 95%. Tachycardia > 100, bradycardia < 60, tachypnea > 20, hypotension < 90/60.",
    "Complete blood count normal ranges: WBC 4,500-11,000/µL, Hemoglobin 12-16 g/dL (women) / 13.5-17.5 (men), Platelets 150,000-400,000/µL, MCV 80-100 fL.",
    "Basic metabolic panel normal ranges: Sodium 135-145 mEq/L, Potassium 3.5-5.0 mEq/L, Chloride 96-106 mEq/L, Bicarbonate 22-28 mEq/L, BUN 7-20 mg/dL, Creatinine 0.7-1.3 mg/dL, Glucose 70-100 mg/dL.",
    "Liver function tests: ALT 7-56 U/L, AST 10-40 U/L, ALP 44-147 U/L, Total bilirubin 0.1-1.2 mg/dL, Albumin 3.5-5.0 g/dL. AST:ALT > 2 suggests alcoholic liver disease.",
    "Arterial blood gas interpretation: pH 7.35-7.45, pCO2 35-45 mmHg, HCO3 22-26 mEq/L. Respiratory acidosis: low pH, high pCO2. Metabolic acidosis: low pH, low HCO3. Winter's formula for expected compensation.",
    "Thyroid function: TSH 0.4-4.0 mIU/L, Free T4 0.8-1.8 ng/dL. High TSH + low T4 = hypothyroidism. Low TSH + high T4 = hyperthyroidism. Subclinical: abnormal TSH with normal T4.",
    "Coagulation studies: PT 11-13.5 seconds, INR 0.8-1.1 (target 2-3 on warfarin), PTT 25-35 seconds. Elevated PT: warfarin, liver disease, vitamin K deficiency. Elevated PTT: heparin, hemophilia.",
    "HbA1c correlation: 6% ≈ avg glucose 126, 7% ≈ 154, 8% ≈ 183, 9% ≈ 212, 10% ≈ 240 mg/dL. Target < 7% for most diabetics, < 8% for elderly with comorbidities.",
    # ── Clinical Decision Making ──
    "Evidence-based medicine hierarchy (strongest to weakest): systematic reviews/meta-analyses → RCTs → cohort studies → case-control studies → case series → expert opinion.",
    "Diagnostic test interpretation: Sensitivity = TP/(TP+FN) — rules OUT disease (SnNout). Specificity = TN/(TN+FP) — rules IN disease (SpPin). High sensitivity for screening, high specificity for confirmation.",
    "Number needed to treat (NNT) = 1/ARR. NNT of 10 means treating 10 patients prevents 1 adverse event. Lower NNT = more effective treatment. NNH (number needed to harm) assesses safety.",
    "Clinical prediction rules: Wells score (DVT/PE), HEART score (ACS risk), CURB-65 (pneumonia severity), CHADS-VASc (stroke risk in AFib), MELD (liver disease severity), Glasgow Coma Scale (neurological assessment).",
    # ── Public Health & Prevention ──
    "Adult immunization: annual influenza, Tdap every 10 years, shingles (Shingrix) at age 50+, pneumococcal (PCV20) at 65+. COVID-19 vaccination per current guidelines.",
    "Cancer screening summary: breast (mammography 40-75), cervical (Pap/HPV 21-65), colorectal (colonoscopy 45-75), lung (LDCT 50-80 with smoking history), prostate (shared decision-making).",
    "Cardiovascular risk assessment: ACC/AHA pooled cohort equations estimate 10-year ASCVD risk. Statin therapy recommended if risk >= 7.5% or LDL >= 190 mg/dL. Lifestyle modifications: diet, exercise, smoking cessation.",
    "Social determinants of health: economic stability, education, healthcare access, neighborhood/environment, social/community context. These factors account for 30-55% of health outcomes, more than clinical care alone.",
    # ── Medical Ethics ──
    "Four principles of medical ethics: Autonomy (patient's right to decide), Beneficence (do good), Non-maleficence (do no harm), Justice (fair distribution of resources). Informed consent requires capacity, disclosure, understanding, voluntariness.",
    "HIPAA: protects patient health information (PHI). Minimum necessary standard — share only information needed. Patients have right to access their records. Breaches require notification within 60 days.",
    "End-of-life care: advance directives (living will, healthcare proxy). POLST for seriously ill patients. Palliative care focuses on symptom management and quality of life. Hospice for prognosis < 6 months.",
    # ── Medical Reasoning ──
    "Differential diagnosis approach: start broad, then narrow using history, physical exam, and investigations. Consider: most common cause, most dangerous cause (don't miss), most treatable cause.",
    "Clinical reasoning pitfalls: anchoring bias (fixating on initial diagnosis), confirmation bias (seeking only supporting evidence), availability bias (overweighting recent cases), premature closure (stopping search too early).",
    "SOAP note format: Subjective (patient's symptoms), Objective (vitals, exam, labs), Assessment (diagnosis/differential), Plan (treatment/follow-up). Standard documentation format in healthcare.",
    "Red flags in clinical presentation: unexplained weight loss, night sweats, persistent fever — think malignancy or chronic infection. New headache in elderly — consider temporal arteritis (check ESR, start steroids empirically).",
    # ── Surgery Basics ──
    "Preoperative assessment: ASA physical status classification (I-VI). NPO guidelines: clear liquids 2 hours, light meal 6 hours, full meal 8 hours before surgery. Anticoagulant management varies by procedure risk.",
    "Surgical site infection prevention: appropriate antibiotic prophylaxis within 60 minutes of incision (cefazolin for most procedures), normothermia, euglycemia, skin preparation with chlorhexidine-alcohol.",
    "Acute abdomen: peritonitis signs (guarding, rigidity, rebound tenderness). Differential: appendicitis (RLQ), cholecystitis (RUQ), diverticulitis (LLQ), perforated ulcer (epigastric). CT abdomen for diagnosis.",
    "Appendicitis: periumbilical pain migrating to RLQ, McBurney's point tenderness, Rovsing's sign. Alvarado score for clinical assessment. CT scan confirms. Treatment: appendectomy (laparoscopic preferred).",
    # ── Geriatrics ──
    "Polypharmacy in elderly (>= 5 medications) increases adverse drug reactions, falls, and hospitalizations. Beers criteria identifies potentially inappropriate medications for older adults. Deprescribing improves outcomes.",
    "Falls in elderly: assess gait, balance, vision, medications, home hazards. Vitamin D supplementation, exercise programs, and home modifications reduce fall risk by 20-30%. Hip fractures have 20% one-year mortality.",
    "Delirium vs dementia: delirium is acute onset, fluctuating consciousness, usually reversible. Dementia is chronic, progressive cognitive decline. Common delirium causes: infection, medications, metabolic disturbances, pain.",
    "Alzheimer's disease: most common dementia (60-80%). Progressive memory loss, disorientation, language difficulty. Treatment: cholinesterase inhibitors (donepezil) for mild-moderate, memantine for moderate-severe. No cure currently.",
]
