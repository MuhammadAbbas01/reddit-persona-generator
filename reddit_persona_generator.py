from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file
import os # Ensure os module is imported for os.getenv
import time # Import time module for sleeping
import random # For randomizing sleep slightly

# DEBUG: This line confirms if the environment variable is loaded. Keep it for now.
print(f"DEBUG: REDDIT_CLIENT_ID from .env: {os.getenv('REDDIT_CLIENT_ID')}")

#!/usr/bin/env python3
"""
Reddit User Persona Generator
A comprehensive script to analyze Reddit user profiles and generate detailed user personas.
This version prioritizes robust, rule-based persona generation with enhanced specificity,
especially for demographics, to ensure a detailed output even without active LLM API access.
"""

import praw
import json
import re
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Any, Optional
import requests
from dataclasses import dataclass
from collections import Counter, defaultdict

# Ensure NLTK is downloaded for sentiment analysis
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except ImportError:
    print("NLTK not installed. Please install it using 'pip install nltk'.")
    exit()
except Exception as e:
    print(f"Warning: NLTK download failed - {e}. Sentiment analysis might be affected.")

import asyncio

@dataclass
class UserData:
    """Data class to store user information"""
    username: str
    posts: List[Dict]
    comments: List[Dict]
    account_age: int
    karma: Dict[str, int]
    subreddits: List[str]

class RedditScraper:
    """Handles Reddit API interactions and data scraping"""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """Initialize Reddit API client"""
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    
    def extract_username(self, url: str) -> str:
        """Extract username from Reddit profile URL"""
        patterns = [
            r'reddit\.com/user/([^/]+)',
            r'reddit\.com/u/([^/]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Invalid Reddit URL format: {url}")
    
    def scrape_user_data(self, username: str, limit: int = 100) -> UserData:
        """Scrape user posts and comments"""
        try:
            user = self.reddit.redditor(username)
            
            # Get user basic info
            account_age = (datetime.now() - datetime.fromtimestamp(user.created_utc)).days
            karma = {
                'post': user.link_karma,
                'comment': user.comment_karma
            }
            
            # Scrape posts
            posts = []
            print(f"  Scraping up to {limit} posts...")
            for i, post in enumerate(user.submissions.new(limit=limit)):
                post_content = post.selftext if post.is_self else ""
                posts.append({
                    'title': post.title,
                    'content': post_content,
                    'subreddit': str(post.subreddit),
                    'score': post.score,
                    'created_utc': post.created_utc,
                    'url': post.url,
                    'id': f"t3_{post.id}",
                    'type': 'post'
                })
                if (i + 1) % 20 == 0:
                    print(f"    Scraped {i + 1} posts...")
            
            # Scrape comments
            comments = []
            print(f"  Scraping up to {limit} comments...")
            for i, comment in enumerate(user.comments.new(limit=limit)):
                comments.append({
                    'content': comment.body,
                    'subreddit': str(comment.subreddit),
                    'score': comment.score,
                    'created_utc': comment.created_utc,
                    'id': f"t1_{comment.id}",
                    'parent_id': comment.parent_id,
                    'type': 'comment'
                })
                if (i + 1) % 20 == 0:
                    print(f"    Scraped {i + 1} comments...")
            
            # Get most active subreddits
            subreddit_counter = Counter()
            
            for post in posts:
                subreddit_counter[post['subreddit']] += 1
            for comment in comments:
                subreddit_counter[comment['subreddit']] += 1
            
            subreddits = [sub for sub, count in subreddit_counter.most_common(20)]
            
            return UserData(
                username=username,
                posts=posts,
                comments=comments,
                account_age=account_age,
                karma=karma,
                subreddits=subreddits
            )
            
        except praw.exceptions.APIException as e:
            raise Exception(f"Reddit API Error scraping user data for {username}: {e}")
        except Exception as e:
            raise Exception(f"General Error scraping user data for {username}: {e}")

class PersonaAnalyzer:
    """Analyzes user data and generates persona using LLM (Gemini) or rule-based fallback"""
    
    def __init__(self):
        """Initialize components for persona analysis"""
        self.sia = SentimentIntensityAnalyzer()
        self.gemini_api_key = os.getenv('GEMINI_API_KEY', '')
        if not self.gemini_api_key:
            print("Warning: GEMINI_API_KEY environment variable is not set. LLM inference will be skipped and rule-based persona will be generated.")

    def analyze_activity_patterns(self, user_data: UserData) -> Dict[str, Any]:
        """Analyze user's posting patterns and activity"""
        total_posts = len(user_data.posts)
        total_comments = len(user_data.comments)
        
        if total_posts + total_comments > 0:
            avg_posts_per_day = (total_posts + total_comments) / max(user_data.account_age, 1)
        else:
            avg_posts_per_day = 0
        
        posting_hours = []
        for post in user_data.posts:
            hour = datetime.fromtimestamp(post['created_utc']).hour
            posting_hours.append(hour)
        
        for comment in user_data.comments:
            hour = datetime.fromtimestamp(comment['created_utc']).hour
            posting_hours.append(hour)
        
        if posting_hours:
            hour_counter = Counter(posting_hours)
            most_active_hours = hour_counter.most_common(3)
        else:
            most_active_hours = []
        
        return {
            'total_posts': total_posts,
            'total_comments': total_comments,
            'avg_posts_per_day': round(avg_posts_per_day, 2),
            'most_active_hours': most_active_hours,
            'account_age_days': user_data.account_age,
            'karma': user_data.karma
        }
    
    def analyze_interests(self, user_data: UserData) -> Dict[str, Any]:
        """Analyze user's interests based on subreddits and content"""
        top_subreddits = user_data.subreddits[:10]
        
        categories = {
            'Technology': ['programming', 'technology', 'coding', 'MachineLearning', 'artificial', 'compsci', 'dev', 'software', 'visionpro', 'chatgpt', 'h1b'],
            'Gaming': ['gaming', 'games', 'Steam', 'PlayStation', 'Xbox', 'Nintendo', 'pcgaming', 'civ5', 'manorlords', 'warriors'],
            'Lifestyle': ['fitness', 'food', 'cooking', 'fashion', 'travel', 'health', 'personalfinance', 'asknyc', 'foodnyc'],
            'Entertainment': ['movies', 'music', 'books', 'television', 'Netflix', 'anime', 'art', 'onepiece'],
            'News & Current Events': ['news', 'worldnews', 'politics', 'economics', 'current events'],
            'Educational & Learning': ['todayilearned', 'explainlikeimfive', 'science', 'history', 'askscience', 'learnprogramming'],
            'Social & Community': ['relationships', 'dating', 'socialskills', 'friendship', 'askreddit', 'casualconversation', 'genz'],
            'Humor & Memes': ['funny', 'memes', 'jokes', 'dankmemes'],
            'Hobbies & Crafts': ['DIY', 'crafts', 'photography', 'woodworking', 'gardening'],
            'Sports': ['sports', 'football', 'basketball', 'soccer', 'cricket', 'nba']
        }
        
        interest_categories = []
        for category, keywords in categories.items():
            for subreddit in top_subreddits:
                if any(keyword.lower() in subreddit.lower() for keyword in keywords):
                    interest_categories.append(category)
                    break
        
        all_content = " ".join([p['title'] + " " + p['content'] for p in user_data.posts if p['content']]) + \
                      " ".join([c['content'] for c in user_data.comments if c['content']])
        
        content_inferred_interests = []
        for category, keywords in categories.items():
            if any(re.search(r'\b' + keyword + r'\b', all_content, re.IGNORECASE) for keyword in keywords):
                content_inferred_interests.append(category)

        all_inferred_interests = list(set(interest_categories + content_inferred_interests))

        return {
            'top_subreddits': top_subreddits,
            'interest_categories': all_inferred_interests,
            'total_subreddits': len(user_data.subreddits)
        }
    
    def analyze_sentiment(self, user_data: UserData) -> Dict[str, Any]:
        """Analyze sentiment of user's posts and comments"""
        all_text = []
        
        for post in user_data.posts:
            if post['content']:
                all_text.append(post['content'])
            all_text.append(post['title'])
        
        for comment in user_data.comments:
            if comment['content']:
                all_text.append(comment['content'])
        
        if not all_text:
            return {'overall_sentiment': 'neutral', 'sentiment_scores': {}}
        
        sentiments = []
        for text in all_text:
            score = self.sia.polarity_scores(text)
            sentiments.append(score)
        
        avg_sentiment = {
            'pos': sum(s['pos'] for s in sentiments) / len(sentiments),
            'neg': sum(s['neg'] for s in sentiments) / len(sentiments),
            'neu': sum(s['neu'] for s in sentiments) / len(sentiments),
            'compound': sum(s['compound'] for s in sentiments) / len(sentiments)
        }
        
        if avg_sentiment['compound'] >= 0.05:
            overall_sentiment = 'positive'
        elif avg_sentiment['compound'] <= -0.05:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_scores': avg_sentiment
        }
    
    async def generate_persona_with_llm(self, user_data: UserData, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed persona using LLM (Gemini API) with structured citations, with fallback."""
        
        # Prepare context for LLM
        context = self.prepare_context(user_data, analysis)
        
        # Define the prompt for the LLM
        prompt = f"""
        **VERY IMPORTANT: For every single characteristic or detail you infer, you MUST provide specific citations in the form of a list of Reddit post or comment IDs that directly support your conclusion. If a characteristic is a general inference not tied to a specific ID (e.g., "active user"), the 'citations' list should contain "General Inference". Do NOT leave 'citations' empty unless no inference is made.**

        Based on the following Reddit user data and analysis, create a comprehensive user persona.
        The persona should include demographics, interests, personality traits, goals, and pain points,
        communication style, online behavior, brand preferences, and content preferences.
        Be concise and direct in your descriptions.

        User Data Context:
        {context}

        Please generate a detailed user persona in the following JSON format.

        {{
            "demographics": {{
                "age_range": {{ "value": "estimated age range (e.g., 25-35)", "citations": [] }},
                "location": {{ "value": "estimated location if mentioned", "citations": [] }},
                "occupation": {{ "value": "estimated occupation/field", "citations": [] }},
                "education_level": {{ "value": "estimated education level", "citations": [] }}
            }},
            "personality_traits": [
                {{ "trait": "list of personality traits", "explanation": "explanation", "citations": [] }}
            ],
            "interests_and_hobbies": [
                {{ "interest": "detailed list of interests and hobbies", "citations": [] }}
            ],
            "goals_and_motivations": [
                {{ "goal": "user's likely goals and motivations", "citations": [] }}
            ],
            "pain_points_and_challenges": [
                {{ "pain_point": "identified pain points and challenges", "citations": [] }}
            ],
            "communication_style": {{ "description": "description of communication style", "citations": [] }},
            "online_behavior": {{ "description": "description of online behavior patterns", "citations": [] }},
            "brand_preferences": [
                {{ "brand": "likely brand preferences based on activity", "citations": [] }}
            ],
            "content_preferences": [
                {{ "preference": "preferred types of content and media", "citations": [] }}
            ]
        }}
        """
        
        # Define the structured response schema for the Gemini API
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "demographics": {
                    "type": "OBJECT",
                    "properties": {
                        "age_range": {"type": "OBJECT", "properties": {"value": {"type": "STRING"}, "citations": {"type": "ARRAY", "items": {"type": "STRING"}}}},
                        "location": {"type": "OBJECT", "properties": {"value": {"type": "STRING"}, "citations": {"type": "ARRAY", "items": {"type": "STRING"}}}},
                        "occupation": {"type": "OBJECT", "properties": {"value": {"type": "STRING"}, "citations": {"type": "ARRAY", "items": {"type": "STRING"}}}},
                        "education_level": {"type": "OBJECT", "properties": {"value": {"type": "STRING"}, "citations": {"type": "ARRAY", "items": {"type": "STRING"}}}}
                    }
                },
                "personality_traits": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "trait": {"type": "STRING"},
                            "explanation": {"type": "STRING"},
                            "citations": {"type": "ARRAY", "items": {"type": "STRING"}}
                        }
                    }
                },
                "interests_and_hobbies": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "interest": {"type": "STRING"},
                            "citations": {"type": "ARRAY", "items": {"type": "STRING"}}
                        }
                    }
                },
                "goals_and_motivations": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "goal": {"type": "STRING"},
                            "citations": {"type": "ARRAY", "items": {"type": "STRING"}}
                        }
                    }
                },
                "pain_points_and_challenges": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "pain_point": {"type": "STRING"},
                            "citations": {"type": "ARRAY", "items": {"type": "STRING"}}
                        }
                    }
                },
                "communication_style": {"type": "OBJECT", "properties": {"description": {"type": "STRING"}, "citations": {"type": "ARRAY", "items": {"type": "STRING"}}}},
                "online_behavior": {"type": "OBJECT", "properties": {"description": {"type": "STRING"}, "citations": {"type": "ARRAY", "items": {"type": "STRING"}}}},
                "brand_preferences": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "brand": {"type": "STRING"},
                            "citations": {"type": "ARRAY", "items": {"type": "STRING"}}
                        }
                    }
                },
                "content_preferences": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "preference": {"type": "STRING"},
                            "citations": {"type": "ARRAY", "items": {"type": "STRING"}}
                        }
                    }
                }
            },
            "propertyOrdering": [
                "demographics", "personality_traits", "interests_and_hobbies",
                "goals_and_motivations", "pain_points_and_challenges",
                "communication_style", "online_behavior", "brand_preferences",
                "content_preferences"
            ]
        }

        # Construct the payload for the Gemini API call
        chatHistory = []
        chatHistory.append({ "role": "user", "parts": [{ "text": prompt }] })
        payload = {
            "contents": chatHistory,
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": response_schema
            }
        }
        
        api_key = self.gemini_api_key
        if not api_key:
            print("Warning: GEMINI_API_KEY environment variable is not set. LLM calls will fail.")
            return self.create_fallback_persona(user_data, analysis)

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

        max_retries = 5 # Increased retries for Gemini
        base_delay = 10 # Longer base delay for Gemini
        
        for attempt in range(max_retries):
            try:
                print(f"  Attempting Gemini LLM call for full persona (attempt {attempt + 1}/{max_retries})...")
                response = requests.post(api_url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload), timeout=180) # Even longer timeout
                response.raise_for_status()
                result = response.json()

                if result.get('candidates') and len(result['candidates']) > 0 and \
                   result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts') and \
                   len(result['candidates'][0]['content']['parts']) > 0:
                    
                    persona_text = result['candidates'][0]['content']['parts'][0]['text']
                    persona_data = json.loads(persona_text)
                    print(f"  Gemini LLM call successful on attempt {attempt + 1}.")
                    return persona_data
                else:
                    print(f"Gemini LLM response did not contain expected data structure on attempt {attempt + 1}. Raw: {json.dumps(result)[:200]}...")
                    return self.create_fallback_persona(user_data, analysis)

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                if status_code == 429 or status_code >= 500:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 7) # Exponential backoff with more jitter
                    print(f"Gemini LLM API call failed with status {status_code}. Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})....")
                    time.sleep(delay)
                else:
                    print(f"Error making Gemini LLM API call: {e}. Falling back to basic persona.")
                    return self.create_fallback_persona(user_data, analysis)
            except requests.exceptions.Timeout:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 7)
                print(f"Gemini LLM API call timed out. Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
            except json.JSONDecodeError as e:
                print(f"Error parsing Gemini LLM JSON response: {e}. Raw: {persona_text[:200]}... Falling back to basic persona.")
                return self.create_fallback_persona(user_data, analysis)
            except Exception as e:
                print(f"An unexpected error occurred during Gemini LLM inference: {e}. Falling back to basic persona.")
                return self.create_fallback_persona(user_data, analysis)
        
        print(f"Max retries reached for Gemini LLM call. Skipping LLM inference and using rule-based persona.")
        return self.create_fallback_persona(user_data, analysis)

    def prepare_context(self, user_data: UserData, analysis: Dict[str, Any]) -> str:
        """Prepare context string for LLM"""
        # Include more data for LLM to draw specific citations from
        recent_posts_str = "\n".join([
            f"  - Post ID: t3_{p['id']} (Subreddit: {p['subreddit']}, Title: \"{p['title'][:100]}...\"): {p['content'][:200]}..."
            for p in user_data.posts if p['content'] or p['title']
        ])
        if not recent_posts_str:
            recent_posts_str = "  No recent posts with content."

        recent_comments_str = "\n".join([
            f"  - Comment ID: t1_{c['id']} (Subreddit: {c['subreddit']}): {c['content'][:200]}..."
            for c in user_data.comments if c['content']
        ])
        if not recent_comments_str:
            recent_comments_str = "  No recent comments with content."
        
        context = f"""
Username: {user_data.username}
Account Age: {analysis['activity']['account_age_days']} days
Karma (Post/Comment): {user_data.karma['post']}/{user_data.karma['comment']}

Activity Analysis:
- Total Posts: {analysis['activity']['total_posts']}
- Total Comments: {analysis['activity']['total_comments']}
- Average Posts per Day: {analysis['activity']['avg_posts_per_day']}
- Most Active Hours (UTC, Hour:Count): {', '.join([f'{h[0]}: {h[1]}' for h in analysis['activity']['most_active_hours']]) if analysis['activity']['most_active_hours'] else 'N/A'}

Top Subreddits: {', '.join(analysis['interests']['top_subreddits']) if analysis['interests']['top_subreddits'] else 'N/A'}
Inferred Interest Categories: {', '.join(analysis['interests']['interest_categories']) if analysis['interests']['interest_categories'] else 'N/A'}

Sentiment Analysis:
- Overall Sentiment: {analysis['sentiment']['overall_sentiment']}

All Scraped Posts (with IDs and content preview):
{recent_posts_str}

All Scraped Comments (with IDs and content preview):
{recent_comments_str}
"""
        return context
    
    def create_fallback_persona(self, user_data: UserData, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a more specific rule-based persona if LLM fails or API key is missing.
        This aims to provide more detail than "Unknown" for demographics.
        """
        # Initialize with defaults
        demographics_data = {
            "age_range": {"value": "Unknown", "citations": ["General Inference"]},
            "location": {"value": "Unknown", "citations": ["General Inference"]},
            "occupation": {"value": "Unknown", "citations": ["General Inference"]},
            "education_level": {"value": "Unknown", "citations": ["General Inference"]}
        }

        # --- Enhanced Rule-Based Inferences for Demographics ---
        # Age inference based on account age
        if user_data.account_age > (365 * 5): # More than 5 years old
            demographics_data["age_range"]["value"] = "Likely Adult (25+ years old)"
            demographics_data["age_range"]["citations"] = ["Account Age"]
        elif user_data.account_age > (365 * 2): # More than 2 years old
            demographics_data["age_range"]["value"] = "Likely Young Adult (20+ years old)"
            demographics_data["age_range"]["citations"] = ["Account Age"]
        
        # Location inference from subreddits or content
        top_subreddits_lower = [s.lower() for s in analysis['interests']['top_subreddits']]
        all_content_lower = " ".join([p['title'].lower() + " " + p['content'].lower() for p in user_data.posts if p['content']]) + \
                            " ".join([c['content'].lower() for c in user_data.comments if c['content']])

        if "asknyc" in top_subreddits_lower or "nyc" in all_content_lower:
            demographics_data["location"]["value"] = "New York City area"
            demographics_data["location"]["citations"] = ["Top Subreddits", "User Content"]
        elif "london" in top_subreddits_lower or "london" in all_content_lower:
            demographics_data["location"]["value"] = "London area"
            demographics_data["location"]["citations"] = ["Top Subreddits", "User Content"]
        # Add more location keywords as needed

        # Occupation inference from subreddits/keywords
        tech_keywords = ['programming', 'compsci', 'dev', 'software', 'h1b', 'machinelearning', 'chatgpt', 'visionpro']
        finance_keywords = ['personalfinance', 'investing']
        gaming_keywords = ['gaming', 'pcgaming', 'civ5', 'manorlords', 'warriors']

        if any(kw in top_subreddits_lower for kw in tech_keywords) or any(re.search(r'\b' + kw + r'\b', all_content_lower) for kw in tech_keywords):
            demographics_data["occupation"]["value"] = "Tech/Software Professional"
            demographics_data["occupation"]["citations"] = ["Top Subreddits", "User Content"]
        elif any(kw in top_subreddits_lower for kw in finance_keywords) or any(re.search(r'\b' + kw + r'\b', all_content_lower) for kw in finance_keywords):
            demographics_data["occupation"]["value"] = "Finance/Business Professional"
            demographics_data["occupation"]["citations"] = ["Top Subreddits", "User Content"]
        elif any(kw in top_subreddits_lower for kw in gaming_keywords) or any(re.search(r'\b' + kw + r'\b', all_content_lower) for kw in gaming_keywords):
            demographics_data["occupation"]["value"] = "Gamer/Gaming Enthusiast"
            demographics_data["occupation"]["citations"] = ["Top Subreddits", "User Content"]


        # --- Rest of the persona (rule-based) ---
        return {
            "demographics": demographics_data,
            "personality_traits": [
                {"trait": "Optimistic" if analysis['sentiment']['overall_sentiment'] == 'positive' else "Neutral/Critical", "explanation": "Based on overall sentiment.", "citations": ["Sentiment Analysis"]},
                {"trait": "Engaged", "explanation": f"Active with {analysis['activity']['total_posts']} posts and {analysis['activity']['total_comments']} comments.", "citations": ["Activity Analysis"]},
                {"trait": "Curious/Diverse", "explanation": f"Participates in {len(analysis['interests']['top_subreddits'])} subreddits.", "citations": ["Top Subreddits"]}
            ],
            "interests_and_hobbies": [
                {"interest": f"Broad interests including: {', '.join(analysis['interests']['interest_categories']) if analysis['interests']['interest_categories'] else 'various topics'}", "citations": ["Interest Categorization"]}
            ],
            "goals_and_motivations": [
                {"goal": "Seeking information and community interaction.", "citations": ["General Inference"]},
                {"goal": "Engaging with topics of interest.", "citations": ["General Inference"]}
            ],
            "pain_points_and_challenges": [
                {"pain_point": "Specific frustrations not directly inferable from public data.", "citations": ["General Inference"]}
            ],
            "communication_style": {"description": f"{analysis['sentiment']['overall_sentiment']} and generally engaging", "citations": ["Sentiment Analysis"]},
            "online_behavior": {"description": f"Posts approximately {analysis['activity']['avg_posts_per_day']} times per day, often during {', '.join([f'{h[0]}h' for h in analysis['activity']['most_active_hours']]) if analysis['activity']['most_active_hours'] else 'various times'}.", "citations": ["Activity Analysis"]},
            "brand_preferences": [
                {"brand": "Not explicitly identifiable from public Reddit data.", "citations": ["General Inference"]}
            ],
            "content_preferences": [
                {"preference": f"Prefers content in diverse subreddits like: {', '.join(analysis['interests']['top_subreddits'][:5])}", "citations": ["Top Subreddits"]}
            ]
        }

class PersonaGenerator:
    """Main class that orchestrates the persona generation process"""
    
    def __init__(self, reddit_config: Dict[str, str]):
        """Initialize the persona generator"""
        self.scraper = RedditScraper(**reddit_config)
        self.analyzer = PersonaAnalyzer()
    
    async def generate_persona(self, profile_url: str, output_file: str = None) -> Dict[str, Any]:
        """Generate complete user persona from Reddit profile URL"""
        try:
            print(f"Starting persona generation for: {profile_url}")
            
            username = self.scraper.extract_username(profile_url)
            print(f"Extracted username: {username}")
            
            print("Scraping user data...")
            user_data = self.scraper.scrape_user_data(username)
            print(f"Scraped {len(user_data.posts)} posts and {len(user_data.comments)} comments")
            
            print("Analyzing user data and generating persona with Gemini LLM (with enhanced fallback)...")
            analysis = {
                'activity': self.analyzer.analyze_activity_patterns(user_data),
                'interests': self.analyzer.analyze_interests(user_data),
                'sentiment': self.analyzer.analyze_sentiment(user_data)
            }
            
            # Attempt to generate persona with LLM, with fallback handled internally by analyzer
            persona = await self.analyzer.generate_persona_with_llm(user_data, analysis)
            
            # Add metadata
            persona['metadata'] = {
                'generated_at': datetime.now().isoformat(),
                'profile_url': profile_url,
                'username': username,
                'analysis_summary': analysis # Keep summary for detailed output
            }
            
            # Save to file
            if output_file:
                self.save_persona_to_file(persona, output_file)
                print(f"Persona saved to: {output_file}")
            
            return persona
            
        except Exception as e:
            print(f"Error generating persona: {str(e)}")
            raise # Re-raise the exception after printing

    def save_persona_to_file(self, persona: Dict[str, Any], filename: str):
        """Save persona to text file in a readable format, extracting embedded citations"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("REDDIT USER PERSONA ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Name and Quote (from image) - Quote is not LLM-driven in this structure, so it's simple
            f.write(f"Name: {persona.get('username', persona['metadata']['username'])}\n") # Use username as name
            f.write(f"Quote: \"An engaged Reddit user, exploring various communities.\"\n") # Default quote
            f.write(f"  Citations: General Inference\n")
            f.write("\n")

            # Helper to write persona sections and extract citations
            def write_section(title, data_dict_or_list):
                f.write(f"{title.upper()}:\n")
                f.write("-" * 20 + "\n")
                
                if isinstance(data_dict_or_list, dict):
                    # Special handling for communication_style and online_behavior
                    if title.lower() in ["communication style", "online behavior"]:
                        description = data_dict_or_list.get('description', '')
                        citations = data_dict_or_list.get('citations', [])
                        f.write(f"Description: {description}\n")
                        if citations:
                            f.write(f"  Citations: {', '.join(citations)}\n")
                    else: # For demographics
                        for key, item in data_dict_or_list.items():
                            value = item.get('value', '')
                            citations = item.get('citations', [])
                            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                            if citations:
                                f.write(f"  Citations: {', '.join(citations)}\n")
                elif isinstance(data_dict_or_list, list):
                    if not data_dict_or_list: # Handle empty lists
                        f.write("No specific inferences found.\n")
                    for item in data_dict_or_list:
                        primary_value = ""
                        if item:
                            # Prioritize 'trait', 'interest', 'goal', 'pain_point', 'brand', 'preference'
                            if 'trait' in item: primary_value = item['trait']
                            elif 'interest' in item: primary_value = item['interest']
                            elif 'goal' in item: primary_value = item['goal']
                            elif 'pain_point' in item: primary_value = item['pain_point']
                            elif 'brand' in item: primary_value = item['brand']
                            elif 'preference' in item: primary_value = item['preference']
                            elif 'value' in item: primary_value = item['value'] # Fallback for simpler structures
                            elif 'description' in item: primary_value = item['description'] # Fallback for simpler structures

                        explanation = item.get('explanation', '')
                        citations = item.get('citations', [])
                        
                        f.write(f"• {primary_value}")
                        if explanation:
                            f.write(f" ({explanation})")
                        f.write("\n")
                        if citations:
                            f.write(f"  Citations: {', '.join(citations)}\n")
                f.write("\n")

            write_section("Demographics", persona.get('demographics', {}))
            write_section("Personality Traits", persona.get('personality_traits', []))
            write_section("Interests and Hobbies", persona.get('interests_and_hobbies', []))
            write_section("Goals and Motivations", persona.get('goals_and_motivations', []))
            write_section("Pain Points and Challenges", persona.get('pain_points_and_challenges', []))
            write_section("Communication Style", persona.get('communication_style', {}))
            write_section("Online Behavior", persona.get('online_behavior', {}))
            write_section("Brand Preferences", persona.get('brand_preferences', []))
            write_section("Content Preferences", persona.get('content_preferences', []))
            
            # Analysis Summary (from metadata)
            f.write("ANALYSIS SUMMARY (For Debugging/Internal Context):\n")
            f.write("-" * 20 + "\n")
            f.write(json.dumps(persona['metadata']['analysis_summary'], indent=2))
            f.write("\n")

async def main():
    """Main function to run the persona generator"""
    parser = argparse.ArgumentParser(description='Generate Reddit User Persona')
    parser.add_argument('profile_url', help='Reddit profile URL')
    parser.add_argument('--output', '-o', default='user_persona.txt', help='Output file path')
    
    args = parser.parse_args()
    
    reddit_config = {
        'client_id': os.getenv('REDDIT_CLIENT_ID'),
        'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
        'user_agent': 'PersonaGenerator/1.0 by Economy-Honeydew-142'
    }
    
    if not all([reddit_config['client_id'], reddit_config['client_secret']]):
        print("Error: REDDIT_CLIENT_ID and/or REDDIT_CLIENT_SECRET environment variables are not set.")
        print("Please ensure your .env file is correctly configured in the same directory as the script.")
        print("You can get these from https://www.reddit.com/prefs/apps (create a 'script' type app).")
        return

    if not os.getenv('GEMINI_API_KEY'): # Check for Gemini API key
        print("Warning: GEMINI_API_KEY environment variable is not set. LLM inference will be skipped, and enhanced rule-based inferences will be used for all sections.")
        print("To enable LLM inference, please set your Gemini API key (e.g., in a .env file or directly in your environment).")
        print("You can get a Gemini API key from Google AI Studio.")

    generator = PersonaGenerator(reddit_config)
    
    try:
        persona = await generator.generate_persona(args.profile_url, args.output)
        print("\nPersona generation completed successfully!")
        print(f"Output saved to: {args.output}")
        
    except Exception as e:
        print(f"\nFatal Error during persona generation: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
