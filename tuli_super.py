import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import wikipedia
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import json
import re
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import sqlite3
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading
import queue
import logging
import time
from duckduckgo_search import ddg
import hashlib

# Initialize core components
app = Flask(__name__)
nlp = spacy.load('en_core_web_sm')
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline('question-answering')
summarizer = pipeline('summarization')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self):
        self.graph = nx.Graph()
        self.embeddings = {}
        self.facts = {}
        self.initialize_db()
    
    def initialize_db(self):
        conn = sqlite3.connect('tuli_knowledge.db')
        c = conn.cursor()
        
        # Create tables for different types of knowledge
        tables = [
            '''CREATE TABLE IF NOT EXISTS facts
               (id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT,
                source TEXT,
                confidence FLOAT,
                embedding BLOB,
                category TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''',
            
            '''CREATE TABLE IF NOT EXISTS relations
               (id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity1 TEXT,
                relation TEXT,
                entity2 TEXT,
                confidence FLOAT,
                source TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''',
            
            '''CREATE TABLE IF NOT EXISTS web_knowledge
               (id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT,
                title TEXT,
                content TEXT,
                summary TEXT,
                embedding BLOB,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP)'''
        ]
        
        for table in tables:
            c.execute(table)
        
        conn.commit()
        conn.close()
    
    def add_fact(self, fact, source, confidence=0.8):
        embedding = sentence_model.encode(fact)
        
        conn = sqlite3.connect('tuli_knowledge.db')
        c = conn.cursor()
        c.execute("INSERT INTO facts (fact, source, confidence, embedding) VALUES (?, ?, ?, ?)",
                 (fact, source, confidence, embedding.tobytes()))
        conn.commit()
        conn.close()
        
        self.facts[fact] = {'source': source, 'confidence': confidence, 'embedding': embedding}
        
        # Extract entities and update knowledge graph
        doc = nlp(fact)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        for ent1 in entities:
            for ent2 in entities:
                if ent1 != ent2:
                    self.graph.add_edge(ent1[0], ent2[0], type=fact)

class LanguageProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.model = AutoModel.from_pretrained('gpt2')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        
    def process_text(self, text):
        # Clean and normalize
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return tokens
    
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    
    def analyze_sentiment(self, text):
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.search_engines = ['google', 'bing', 'duckduckgo']
    
    def scrape_url(self, url):
        try:
            # Enhanced error handling and retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    article = Article(url)
                    article.download()
                    article.parse()
                    article.nlp()
                    
                    # Get main content with enhanced parsing
                    response = requests.get(url, headers=self.headers, timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Remove unwanted elements
                    for element in soup.find_all(['script', 'style', 'nav', 'footer']):
                        element.decompose()
                    
                    # Get text content with better filtering
                    paragraphs = soup.find_all(['p', 'article', 'section', 'div'], class_=lambda x: x and any(c in str(x).lower() for c in ['content', 'article', 'text']))
                    content = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
                    
                    # Enhanced metadata extraction
                    metadata = {
                        'title': article.title or soup.title.string if soup.title else '',
                        'authors': article.authors,
                        'publish_date': article.publish_date,
                        'keywords': article.keywords,
                        'summary': article.summary,
                        'text': content,
                        'language': article.meta_lang,
                        'top_image': article.top_image,
                        'tables': [table.get_text() for table in soup.find_all('table')],
                        'links': [{'text': a.get_text(), 'href': a.get('href')} for a in soup.find_all('a', href=True)]
                    }
                    
                    # Store in database with enhanced data
                    conn = sqlite3.connect('tuli_knowledge.db')
                    c = conn.cursor()
                    c.execute("""
                        INSERT INTO web_knowledge 
                        (url, title, content, summary, embedding) 
                        VALUES (?, ?, ?, ?, ?)
                    """, (url, metadata['title'], content, metadata['summary'], 
                         sentence_model.encode(content[:1000]).tobytes()))
                    conn.commit()
                    conn.close()
                    
                    return metadata
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(1 * (attempt + 1))
                    
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None
    
    def search_and_scrape(self, query, max_results=5):
        """Search the web and scrape relevant results"""
        try:
            # Search Wikipedia
            wiki_results = []
            try:
                wiki_pages = wikipedia.search(query)
                for page_title in wiki_pages[:2]:
                    try:
                        page = wikipedia.page(page_title)
                        wiki_results.append({
                            'title': page.title,
                            'content': page.content[:1000],
                            'url': page.url,
                            'source': 'wikipedia'
                        })
                    except:
                        continue
            except:
                pass
            
            # Use DuckDuckGo API (no key required)
            web_results = ddg(query, max_results=max_results)
            
            all_results = []
            for result in web_results:
                if result.get('link'):
                    scraped_data = self.scrape_url(result['link'])
                    if scraped_data:
                        all_results.append({
                            'title': scraped_data['title'],
                            'content': scraped_data['text'][:1000],
                            'url': result['link'],
                            'source': 'web'
                        })
            
            return wiki_results + all_results
            
        except Exception as e:
            logger.error(f"Error in search_and_scrape: {str(e)}")
            return []

class AutonomousLearner:
    def __init__(self):
        self.web_scraper = WebScraper()
        self.knowledge_base = KnowledgeBase()
        self.learning_thread = None
        self.is_learning = False
        self.learning_queue = queue.Queue()
        self.topics_to_learn = set()
    
    def start_learning(self):
        """Start autonomous learning process"""
        if not self.is_learning:
            self.is_learning = True
            self.learning_thread = threading.Thread(target=self._learning_loop)
            self.learning_thread.daemon = True
            self.learning_thread.start()
    
    def _learning_loop(self):
        """Continuous learning loop"""
        while self.is_learning:
            try:
                # Get topics from conversation history
                self._extract_learning_topics()
                
                # Process each topic
                for topic in self.topics_to_learn:
                    if topic not in self.learning_queue.queue:
                        self.learning_queue.put(topic)
                
                # Process learning queue
                while not self.learning_queue.empty():
                    topic = self.learning_queue.get()
                    self._learn_topic(topic)
                    self.learning_queue.task_done()
                
                time.sleep(60)  # Wait before next learning cycle
                
            except Exception as e:
                logger.error(f"Error in learning loop: {str(e)}")
                time.sleep(10)
    
    def _extract_learning_topics(self):
        """Extract topics to learn from conversation history"""
        try:
            conn = sqlite3.connect('tuli_knowledge.db')
            c = conn.cursor()
            c.execute("SELECT input FROM conversation_history ORDER BY timestamp DESC LIMIT 100")
            recent_conversations = c.fetchall()
            conn.close()
            
            for conv in recent_conversations:
                # Extract entities and key phrases
                doc = nlp(conv[0])
                entities = [ent.text for ent in doc.ents]
                key_phrases = self._extract_key_phrases(conv[0])
                
                self.topics_to_learn.update(entities)
                self.topics_to_learn.update(key_phrases)
                
        except Exception as e:
            logger.error(f"Error extracting learning topics: {str(e)}")
    
    def _extract_key_phrases(self, text):
        """Extract key phrases from text"""
        try:
            # Use TextBlob for noun phrase extraction
            blob = TextBlob(text)
            noun_phrases = blob.noun_phrases
            
            # Use spaCy for additional phrase extraction
            doc = nlp(text)
            phrases = []
            
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) > 1:  # Only multi-word phrases
                    phrases.append(chunk.text)
            
            return set(noun_phrases + phrases)
        except Exception as e:
            logger.error(f"Error extracting key phrases: {str(e)}")
            return set()
    
    def _learn_topic(self, topic):
        """Learn about a specific topic"""
        try:
            # Search and scrape information
            results = self.web_scraper.search_and_scrape(topic)
            
            # Process and store information
            for result in results:
                # Extract facts and relationships
                doc = nlp(result['content'])
                facts = self._extract_facts(doc)
                
                # Store in knowledge base
                for fact in facts:
                    self.knowledge_base.add_fact(
                        fact=fact['text'],
                        source=result['url'],
                        confidence=fact['confidence']
                    )
                
                # Extract and store relationships
                relationships = self._extract_relationships(doc)
                for rel in relationships:
                    self.knowledge_base.add_relationship(
                        entity1=rel['entity1'],
                        relation=rel['relation'],
                        entity2=rel['entity2'],
                        confidence=rel['confidence'],
                        source=result['url']
                    )
            
        except Exception as e:
            logger.error(f"Error learning topic {topic}: {str(e)}")
    
    def _extract_facts(self, doc):
        """Extract facts from text"""
        facts = []
        for sent in doc.sents:
            # Check if sentence contains important entities
            if len([ent for ent in sent.ents]) > 0:
                # Calculate confidence based on sentence structure
                confidence = self._calculate_fact_confidence(sent)
                if confidence > 0.5:
                    facts.append({
                        'text': sent.text,
                        'confidence': confidence
                    })
        return facts
    
    def _extract_relationships(self, doc):
        """Extract relationships between entities"""
        relationships = []
        for sent in doc.sents:
            entities = list(sent.ents)
            if len(entities) >= 2:
                # Find verbs between entities
                for i in range(len(entities) - 1):
                    verb = self._find_connecting_verb(sent, entities[i], entities[i + 1])
                    if verb:
                        relationships.append({
                            'entity1': entities[i].text,
                            'relation': verb,
                            'entity2': entities[i + 1].text,
                            'confidence': 0.7
                        })
        return relationships
    
    def _calculate_fact_confidence(self, sent):
        """Calculate confidence score for a potential fact"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence for sentences with named entities
        confidence += 0.1 * len([ent for ent in sent.ents])
        
        # Increase confidence for sentences with specific patterns
        if any(token.dep_ == 'ROOT' and token.pos_ == 'VERB' for token in sent):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _find_connecting_verb(self, sent, ent1, ent2):
        """Find verb connecting two entities"""
        for token in sent:
            if token.pos_ == 'VERB' and ent1.start_char < token.idx < ent2.start_char:
                return token.lemma_
        return None

class AdvancedKnowledgeProcessor:
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.language_processor = LanguageProcessor()
        self.web_scraper = WebScraper()
        self.conversation_history = []
        self.cache = {}
        self.cache_timeout = 3600
        self.autonomous_learner = AutonomousLearner()
        self.context_window = 10  # Number of previous messages to consider
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        
    def process_input(self, user_input):
        # Extract key information
        doc = nlp(user_input)
        entities = [ent.text for ent in doc.ents]
        key_phrases = self._extract_key_phrases(user_input)
        sentiment = self.language_processor.analyze_sentiment(user_input)
        
        # Get context from conversation history
        context = self._get_context()
        
        # Search knowledge base with improved relevance
        relevant_facts = self._search_knowledge_base(user_input, entities, key_phrases)
        
        # Get web information with caching
        web_info = self._get_web_information(user_input, entities, key_phrases)
        
        # Combine all information sources
        combined_info = self._combine_information(user_input, relevant_facts, web_info, context, sentiment)
        
        # Generate response
        response = self._generate_response(combined_info, sentiment)
        
        # Update conversation history
        self._update_history(user_input, response, sentiment)
        
        # Learn from interaction
        self._learn_from_interaction(user_input, response, sentiment)
        
        return response
    
    def _extract_key_phrases(self, text):
        """Extract key phrases using multiple methods"""
        # Use TextBlob for noun phrases
        blob = TextBlob(text)
        noun_phrases = blob.noun_phrases
        
        # Use spaCy for additional phrase extraction
        doc = nlp(text)
        phrases = []
        
        # Extract noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:
                phrases.append(chunk.text)
        
        # Extract verb phrases
        for token in doc:
            if token.pos_ == "VERB":
                verb_phrase = [t.text for t in token.subtree]
                if len(verb_phrase) > 1:
                    phrases.append(" ".join(verb_phrase))
        
        return set(noun_phrases + phrases)
    
    def _get_context(self):
        """Get relevant context from conversation history"""
        context = []
        for i, conv in enumerate(self.conversation_history[-self.context_window:]):
            weight = 1.0 - (0.1 * i)  # More recent messages have higher weight
            context.append({
                'text': conv['input'] + ' ' + conv['response'],
                'weight': weight,
                'sentiment': conv['sentiment']
            })
        return context
    
    def _search_knowledge_base(self, query, entities, key_phrases):
        """Enhanced knowledge base search"""
        results = []
        
        # Search by entities
        for entity in entities:
            entity_facts = self.knowledge_base.search_by_entity(entity)
            results.extend(entity_facts)
        
        # Search by key phrases
        for phrase in key_phrases:
            phrase_facts = self.knowledge_base.search_by_phrase(phrase)
            results.extend(phrase_facts)
        
        # Search by semantic similarity
        semantic_facts = self.knowledge_base.search_by_semantic_similarity(query)
        results.extend(semantic_facts)
        
        # Remove duplicates and sort by confidence
        unique_results = {}
        for fact in results:
            if fact['text'] not in unique_results or fact['confidence'] > unique_results[fact['text']]['confidence']:
                unique_results[fact['text']] = fact
        
        return sorted(unique_results.values(), key=lambda x: x['confidence'], reverse=True)
    
    def _get_web_information(self, query, entities, key_phrases):
        """Enhanced web information gathering"""
        web_info = []
        
        # Search Wikipedia
        try:
            wiki_results = wikipedia.search(query)
            for page_title in wiki_results[:2]:
                try:
                    page = wikipedia.page(page_title)
                    web_info.append({
                        'title': page.title,
                        'content': page.content[:1000],
                        'url': page.url,
                        'source': 'wikipedia',
                        'confidence': 0.8
                    })
                except:
                    continue
        except:
            pass
        
        # Search web
        web_results = self.web_scraper.search_and_scrape(query)
        for result in web_results:
            web_info.append({
                'title': result['title'],
                'content': result['content'],
                'url': result['url'],
                'source': 'web',
                'confidence': 0.6
            })
        
        return web_info
    
    def _combine_information(self, query, facts, web_info, context, sentiment):
        """Combine information from multiple sources"""
        combined = {
            'query': query,
            'facts': facts,
            'web_info': web_info,
            'context': context,
            'sentiment': sentiment,
            'entities': [ent.text for ent in nlp(query).ents],
            'key_phrases': self._extract_key_phrases(query)
        }
        
        # Calculate overall confidence
        combined['confidence'] = self._calculate_confidence(combined)
        
        return combined
    
    def _calculate_confidence(self, combined_info):
        """Calculate overall confidence in the information"""
        confidence = 0.0
        weights = {
            'facts': 0.4,
            'web_info': 0.3,
            'context': 0.2,
            'sentiment': 0.1
        }
        
        # Weight facts by confidence
        if combined_info['facts']:
            fact_confidence = sum(f['confidence'] for f in combined_info['facts']) / len(combined_info['facts'])
            confidence += fact_confidence * weights['facts']
        
        # Weight web info by confidence
        if combined_info['web_info']:
            web_confidence = sum(w['confidence'] for w in combined_info['web_info']) / len(combined_info['web_info'])
            confidence += web_confidence * weights['web_info']
        
        # Weight context by recency
        if combined_info['context']:
            context_confidence = sum(c['weight'] for c in combined_info['context']) / len(combined_info['context'])
            confidence += context_confidence * weights['context']
        
        # Weight sentiment by strength
        sentiment_confidence = abs(combined_info['sentiment']['polarity'])
        confidence += sentiment_confidence * weights['sentiment']
        
        return min(confidence, 1.0)
    
    def _generate_response(self, combined_info, sentiment):
        """Generate response based on combined information"""
        # If confidence is low, ask for clarification
        if combined_info['confidence'] < self.confidence_threshold:
            return "I'm not entirely sure about that. Could you provide more details?"
        
        # Use question-answering if it's a question
        if '?' in combined_info['query']:
            try:
                context_text = ' '.join([f['text'] for f in combined_info['facts']])
                if combined_info['web_info']:
                    context_text += ' ' + ' '.join([w['content'] for w in combined_info['web_info']])
                
                answer = qa_pipeline({
                    'question': combined_info['query'],
                    'context': context_text[:1000]
                })
                
                if answer['score'] > 0.5:
                    return self._format_response(answer['answer'], sentiment)
            except Exception as e:
                logger.error(f"Error in QA pipeline: {str(e)}")
        
        # Use summarization for long responses
        if combined_info['facts'] or combined_info['web_info']:
            try:
                text_to_summarize = ''
                if combined_info['facts']:
                    text_to_summarize += ' '.join([f['text'] for f in combined_info['facts'][:3]])
                if combined_info['web_info']:
                    text_to_summarize += ' ' + ' '.join([w['content'] for w in combined_info['web_info'][:2]])
                
                if len(text_to_summarize) > 100:
                    summary = summarizer(text_to_summarize, 
                                       max_length=150,
                                       min_length=50,
                                       do_sample=False)[0]['summary_text']
                    return self._format_response(summary, sentiment)
            except Exception as e:
                logger.error(f"Error in summarization: {str(e)}")
        
        # Fallback to most relevant fact or web result
        if combined_info['facts']:
            return self._format_response(combined_info['facts'][0]['text'], sentiment)
        elif combined_info['web_info']:
            return self._format_response(combined_info['web_info'][0]['content'][:200], sentiment)
        
        return "I'm still learning about that topic. Could you tell me more?"
    
    def _format_response(self, text, sentiment):
        """Format response based on sentiment and content"""
        # Adjust response tone based on sentiment
        if sentiment['polarity'] < -0.5:
            prefix = "I understand this might be a sensitive topic. "
        elif sentiment['polarity'] > 0.5:
            prefix = "I'm glad you asked! "
        else:
            prefix = ""
        
        return prefix + text.strip()
    
    def _update_history(self, user_input, response, sentiment):
        """Update conversation history"""
        self.conversation_history.append({
            'input': user_input,
            'response': response,
            'sentiment': sentiment,
            'timestamp': datetime.now()
        })
        
        # Keep history within window size
        if len(self.conversation_history) > self.context_window * 2:
            self.conversation_history = self.conversation_history[-self.context_window:]
    
    def _learn_from_interaction(self, user_input, response, sentiment):
        """Learn from the interaction"""
        # Extract new facts
        doc = nlp(user_input)
        facts = self._extract_facts(doc)
        
        # Add new facts to knowledge base
        for fact in facts:
            if fact['confidence'] > self.confidence_threshold:
                self.knowledge_base.add_fact(
                    fact=fact['text'],
                    source='conversation',
                    confidence=fact['confidence']
                )
        
        # Extract and store relationships
        relationships = self._extract_relationships(doc)
        for rel in relationships:
            if rel['confidence'] > self.confidence_threshold:
                self.knowledge_base.add_relationship(
                    entity1=rel['entity1'],
                    relation=rel['relation'],
                    entity2=rel['entity2'],
                    confidence=rel['confidence'],
                    source='conversation'
                )
        
        # Update learning rate based on interaction quality
        if sentiment['polarity'] > 0.5:
            self.learning_rate *= 1.1  # Increase learning rate for positive interactions
        elif sentiment['polarity'] < -0.5:
            self.learning_rate *= 0.9  # Decrease learning rate for negative interactions
    
    def _extract_facts(self, doc):
        """Extract facts from text"""
        facts = []
        for sent in doc.sents:
            # Check if sentence contains important entities
            if len([ent for ent in sent.ents]) > 0:
                # Calculate confidence based on sentence structure
                confidence = self._calculate_fact_confidence(sent)
                if confidence > 0.5:
                    facts.append({
                        'text': sent.text,
                        'confidence': confidence
                    })
        return facts
    
    def _extract_relationships(self, doc):
        """Extract relationships between entities"""
        relationships = []
        for sent in doc.sents:
            entities = list(sent.ents)
            if len(entities) >= 2:
                # Find verbs between entities
                for i in range(len(entities) - 1):
                    verb = self._find_connecting_verb(sent, entities[i], entities[i + 1])
                    if verb:
                        relationships.append({
                            'entity1': entities[i].text,
                            'relation': verb,
                            'entity2': entities[i + 1].text,
                            'confidence': 0.7
                        })
        return relationships
    
    def _calculate_fact_confidence(self, sent):
        """Calculate confidence score for a potential fact"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence for sentences with named entities
        confidence += 0.1 * len([ent for ent in sent.ents])
        
        # Increase confidence for sentences with specific patterns
        if any(token.dep_ == 'ROOT' and token.pos_ == 'VERB' for token in sent):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _find_connecting_verb(self, sent, ent1, ent2):
        """Find verb connecting two entities"""
        for token in sent:
            if token.pos_ == 'VERB' and ent1.start_char < token.idx < ent2.start_char:
                return token.lemma_
        return None

# Initialize advanced knowledge processor
advanced_processor = AdvancedKnowledgeProcessor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = data.get('message', '')
        
        if not user_input:
            return jsonify({
                'error': 'No message provided',
                'status': 'error'
            }), 400
        
        # Process input with advanced processor
        response = advanced_processor.process_input(user_input)
        
        # Get learning status
        learning_status = {
            'is_learning': advanced_processor.autonomous_learner.is_learning,
            'topics_in_queue': advanced_processor.autonomous_learner.learning_queue.qsize(),
            'topics_learned': len(advanced_processor.autonomous_learner.topics_to_learn),
            'learning_rate': advanced_processor.learning_rate,
            'confidence_threshold': advanced_processor.confidence_threshold
        }
        
        # Return enhanced response data
        return jsonify({
            'response': response,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'success',
            'metadata': {
                'sentiment': advanced_processor.conversation_history[-1]['sentiment'],
                'sources': [
                    {'type': 'knowledge_base', 'count': len(advanced_processor.knowledge_base.facts)},
                    {'type': 'web', 'count': len(advanced_processor.cache)}
                ],
                'learning': learning_status
            }
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/learn', methods=['POST'])
def learn():
    data = request.get_json()
    fact = data.get('fact', '')
    source = data.get('source', 'user')
    
    if fact:
        response_generator.knowledge_base.add_fact(fact, source)
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'No fact provided'})

@app.route('/scrape', methods=['POST'])
def scrape():
    data = request.get_json()
    url = data.get('url', '')
    
    if url:
        result = response_generator.web_scraper.scrape_url(url)
        if result:
            return jsonify(result)
    return jsonify({'error': 'Invalid URL or scraping failed'})

if __name__ == '__main__':
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    
    # Initialize autonomous learner
    autonomous_learner = AutonomousLearner()
    
    # Start the Flask server
    app.run(debug=True, host='0.0.0.0', port=5000) 