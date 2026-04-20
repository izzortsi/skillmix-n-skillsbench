## 1. Extracting Skills (Names)

### Method 1: Task-Based Skill Labeling
- Provide task examples to a powerful LLM (within context limits)
- Instruct the LLM to label each example with required skills using a constrained format
  - Example format: "strings of four words connected with underscore"
  - Ensures expressiveness while maintaining structure
- Feed the complete list of skill labels back to the LLM
- Request semantic clustering of these skill labels
- The LLM generates hierarchical skill organization
- *Application*: Used with math datasets (Hendrycks math) to extract skills like "circle_properties_area_calculation"

### Method 2: Direct Elicitation of Novel Skills
- Prompt the LLM with: "Give me a broad skill that has no existing name but which many humans will recognize"
- Example output: "linguistic exorcism" (rewording text to be less offensive/more useful)
- Follow-up prompts to extract subskills:
  - "What are some subskills of [skill name]?"
  - "Give me finer-grained skills associated with [skill name]"
- *Application*: Generated novel skill taxonomies like "synonym_substitution_technique" and "tone_moderation_adjustment"

### Method 3: Comprehensive Skill Catalog Generation
- Instruct the LLM: "You are a great chat agent. Give us instruction-following skills in this format."
- Request hierarchical organization of skills
- Collect approximately 1,000 skills through this process
- *Application*: Created skill taxonomy for instruction-following capabilities used in synthetic data generation

## 2. Extracting Skills from Text (Wikipedia, Corpus Data)

### Method 1: Wikipedia Skill Extraction
- Start with named language skills that appear in Wikipedia with descriptions
- Leverage the fact that "every model in the world has trained on Wikipedia"
- Verify model understanding by testing if models can generate examples of these skills
- *Application*: Used as baseline skills for composition experiments (e.g., "metaphor," "ad_hominem_attack")

### Method 2: Skill-Mix Evaluation Framework
- Create a long list of potential language skills from corpus analysis
- Randomly select skills from this list for evaluation
- Test if models can compose these skills in novel combinations
- Use probability calculations based on estimated training data size to verify novelty
- *Application*: Demonstrated GPT-4 could compose up to 5 randomly selected skills with most generations being novel

### Method 3: Context Analysis for Mini-Theories
- Analyze how LLMs build "mini theories" for text pieces during next-word prediction
- Identify the different foci or elements the model must track to predict the next word
- *Application*: Explains how models develop understanding of multiple concepts within a single text passage

## 3. Generating Skill Examples

### Method 1: Random Skill Pair Composition
- Using the extracted skill catalog, randomly select pairs of skills
- Prompt: "Create a plausible user query and answer exhibiting [skill1] and [skill2]"
- Ensure the generated content is coherent and demonstrates both skills
- *Application*: Generated 4,000 question-answer pairs that enabled Llama3 8B to surpass Opus 405B in instruction-following

### Method 2: Domain-Specific Skill Combination
- For math: "Create a question requiring [math_skill1] and [math_skill2]"
- For instruction-following: "Create a detailed question with many moving parts requiring [critical_thinking] and [communication_literature]"
- *Application*: Produced math questions combining traditionally separate topics (e.g., algebra and probability)

### Method 3: Synthetic Construction Testing
- Prompt: "Give me an example of a piece of text with two sentences, which could appear in a fiction about [topic] and has these two skills"
- *Application*: Tested models' ability to compose skills in novel contexts (e.g., "sushi" and "ad_hominem_attack")
