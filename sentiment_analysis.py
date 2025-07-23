#!/usr/bin/env python3
"""
@treehut Sentiment Analysis using Claude Sonnet 4
Analyzes customer sentiment in Instagram comments for brand reputation insights
"""

import pandas as pd
import numpy as np
import json
import time
import os
from datetime import datetime
import anthropic
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class TreeHutSentimentAnalyzer:
    def __init__(self, csv_path='engagements.csv', api_key=None):
        """Initialize the sentiment analyzer"""
        self.df = pd.read_csv(csv_path)
        self.prepare_data()
        
        # Initialize Claude API
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            # Try to get from environment variable
            try:
                self.client = anthropic.Anthropic()
            except Exception as e:
                print("❌ Error: Please set ANTHROPIC_API_KEY environment variable or pass api_key parameter")
                print("   export ANTHROPIC_API_KEY='your-api-key-here'")
                raise e
        
        print(f"✅ Initialized sentiment analyzer with {len(self.df):,} comments")
    
    def prepare_data(self):
        """Clean and prepare the data"""
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], format='mixed')
        self.df['comment_text'] = self.df['comment_text'].fillna('')
        self.df['media_caption'] = self.df['media_caption'].fillna('')
        
        # Filter out very short comments (likely just emojis or tags)
        self.df = self.df[self.df['comment_text'].str.len() > 5]
        print(f"📊 Filtered to {len(self.df):,} substantive comments for analysis")
    
    def analyze_comment_sentiment(self, comment: str) -> Dict:
        """Analyze sentiment of a single comment using Claude"""
        prompt = f"""
        Analyze the sentiment of this Instagram comment about TreeHut beauty products:
        
        Comment: "{comment}"
        
        Please provide:
        1. Overall sentiment: positive, negative, or neutral
        2. Confidence score: 0.0 to 1.0
        3. Key themes: list of 1-3 themes (e.g., "product_quality", "scent", "texture", "price", "availability")
        4. Specific feedback: any specific praise or complaints
        
        Respond in JSON format:
        {{
            "sentiment": "positive|negative|neutral",
            "confidence": 0.85,
            "themes": ["product_quality", "scent"],
            "feedback": "brief summary of specific feedback"
        }}
        """
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse JSON response
            result = json.loads(response.content[0].text)
            return result
            
        except Exception as e:
            print(f"⚠️ Error analyzing comment: {str(e)[:100]}...")
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "themes": ["error"],
                "feedback": "analysis_failed"
            }
    
    def analyze_sample_comments(self, sample_size: int = 100, random_seed: int = 42) -> pd.DataFrame:
        """Analyze sentiment for a sample of comments, ensuring diverse post coverage"""
        print(f"\n🔍 Analyzing sentiment for {sample_size} sample comments...")

        # Sample comments strategically to get diverse post coverage
        np.random.seed(random_seed)

        # Get top posts by comment count to ensure we analyze high-impact content
        top_posts = self.df.groupby('media_id').size().sort_values(ascending=False).head(20)

        # Sample from top posts (70%) and random posts (30%) for balanced coverage
        top_post_sample_size = int(sample_size * 0.7)
        random_sample_size = sample_size - top_post_sample_size

        # Sample from top posts
        top_posts_df = self.df[self.df['media_id'].isin(top_posts.index)]
        top_sample = top_posts_df.sample(n=min(top_post_sample_size, len(top_posts_df)))

        # Sample from remaining posts
        remaining_df = self.df[~self.df['media_id'].isin(top_posts.index)]
        random_sample = remaining_df.sample(n=min(random_sample_size, len(remaining_df)))

        # Combine samples
        sample_df = pd.concat([top_sample, random_sample]).sample(frac=1)  # Shuffle

        results = []
        for idx, row in sample_df.iterrows():
            print(f"   Processing comment {len(results)+1}/{len(sample_df)}...", end='\r')

            sentiment_result = self.analyze_comment_sentiment(row['comment_text'])

            results.append({
                'media_id': row['media_id'],
                'comment_text': row['comment_text'],
                'media_caption': row['media_caption'][:100] + "..." if len(row['media_caption']) > 100 else row['media_caption'],
                'timestamp': row['timestamp'],
                'sentiment': sentiment_result['sentiment'],
                'confidence': sentiment_result['confidence'],
                'themes': sentiment_result['themes'],
                'feedback': sentiment_result['feedback']
            })

            # Rate limiting - be respectful to the API
            time.sleep(0.5)

        print(f"\n✅ Completed sentiment analysis for {len(results)} comments across {sample_df['media_id'].nunique()} posts")
        return pd.DataFrame(results)
    
    def analyze_by_individual_posts(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment aggregated by individual posts"""
        print("\n📊 Analyzing sentiment by individual posts...")

        post_analysis = []

        for media_id, post_comments in sentiment_df.groupby('media_id'):
            # Get post info
            post_caption = post_comments['media_caption'].iloc[0]
            comment_count = len(post_comments)

            # Calculate sentiment distribution
            sentiment_counts = post_comments['sentiment'].value_counts()
            total_comments = len(post_comments)

            positive_pct = (sentiment_counts.get('positive', 0) / total_comments) * 100
            negative_pct = (sentiment_counts.get('negative', 0) / total_comments) * 100
            neutral_pct = (sentiment_counts.get('neutral', 0) / total_comments) * 100

            # Calculate average confidence
            avg_confidence = post_comments['confidence'].mean()

            # Identify post type
            is_giveaway = 'giveaway' in post_caption.lower() or 'contest' in post_caption.lower() or 'win' in post_caption.lower()
            is_pr_recruitment = 'pr' in post_caption.lower() and ('application' in post_caption.lower() or 'apply' in post_caption.lower())

            # Extract dominant themes
            all_themes = []
            for themes_list in post_comments['themes']:
                if isinstance(themes_list, list):
                    all_themes.extend(themes_list)
            top_themes = pd.Series(all_themes).value_counts().head(3).index.tolist() if all_themes else []

            # Calculate sentiment score (positive % - negative %)
            sentiment_score = positive_pct - negative_pct

            post_analysis.append({
                'media_id': media_id,
                'post_caption_preview': post_caption[:80] + "..." if len(post_caption) > 80 else post_caption,
                'comment_count': comment_count,
                'positive_pct': positive_pct,
                'negative_pct': negative_pct,
                'neutral_pct': neutral_pct,
                'sentiment_score': sentiment_score,
                'avg_confidence': avg_confidence,
                'is_giveaway': is_giveaway,
                'is_pr_recruitment': is_pr_recruitment,
                'top_themes': top_themes,
                'dominant_sentiment': sentiment_counts.index[0] if len(sentiment_counts) > 0 else 'neutral'
            })

        return pd.DataFrame(post_analysis).sort_values('sentiment_score', ascending=False)

    def analyze_by_post_type(self, sentiment_df: pd.DataFrame) -> Dict:
        """Analyze sentiment by post type (giveaway vs regular)"""
        print("\n📊 Analyzing sentiment by post type...")

        # Identify giveaway posts
        giveaway_mask = sentiment_df['media_caption'].str.contains('giveaway|contest|win', case=False, na=False)

        giveaway_sentiment = sentiment_df[giveaway_mask]['sentiment'].value_counts(normalize=True)
        regular_sentiment = sentiment_df[~giveaway_mask]['sentiment'].value_counts(normalize=True)

        return {
            'giveaway_posts': {
                'count': giveaway_mask.sum(),
                'sentiment_distribution': giveaway_sentiment.to_dict()
            },
            'regular_posts': {
                'count': (~giveaway_mask).sum(),
                'sentiment_distribution': regular_sentiment.to_dict()
            }
        }
    
    def extract_themes(self, sentiment_df: pd.DataFrame) -> Dict:
        """Extract and count common themes from sentiment analysis"""
        print("\n🏷️ Extracting common themes...")
        
        all_themes = []
        for themes_list in sentiment_df['themes']:
            if isinstance(themes_list, list):
                all_themes.extend(themes_list)
        
        theme_counts = pd.Series(all_themes).value_counts()
        
        # Group by sentiment
        theme_by_sentiment = {}
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_themes = []
            subset = sentiment_df[sentiment_df['sentiment'] == sentiment]
            for themes_list in subset['themes']:
                if isinstance(themes_list, list):
                    sentiment_themes.extend(themes_list)
            theme_by_sentiment[sentiment] = pd.Series(sentiment_themes).value_counts().head(5).to_dict()
        
        return {
            'overall_themes': theme_counts.head(10).to_dict(),
            'themes_by_sentiment': theme_by_sentiment
        }
    
    def create_sentiment_visualizations(self, sentiment_df: pd.DataFrame, analysis_results: Dict):
        """Create visualizations for sentiment analysis"""
        print("\n📊 Creating sentiment visualizations...")

        # Create visualizations directory
        viz_dir = 'visualizations/brand_reputation'
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
            print(f"📁 Created directory: {viz_dir}/")

        colors = {'positive': 'lightgreen', 'negative': 'lightcoral', 'neutral': 'lightgray'}

        # 1. Overall sentiment distribution
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sentiment_counts = sentiment_df['sentiment'].value_counts()
        bar_colors = [colors.get(sentiment, 'lightblue') for sentiment in sentiment_counts.index]

        bars = ax1.bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors, alpha=0.8)
        ax1.set_title('Overall Comment Sentiment Distribution', fontweight='bold', fontsize=14, pad=20)
        ax1.set_ylabel('Number of Comments')

        # Add percentage labels
        total = sentiment_counts.sum()
        for bar, count in zip(bars, sentiment_counts.values):
            percentage = (count / total) * 100
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        sentiment_dist_path = os.path.join(viz_dir, 'overall_sentiment_distribution.png')
        plt.savefig(sentiment_dist_path, dpi=300, bbox_inches='tight')
        print(f"😊 Overall sentiment chart saved as '{sentiment_dist_path}'")
        plt.close()

        # 2. Individual post sentiment scores
        if 'post_analysis' in analysis_results:
            post_df = analysis_results['post_analysis']

            fig2, ax2 = plt.subplots(figsize=(14, 8))

            # Create scatter plot of posts by sentiment score and comment count
            scatter = ax2.scatter(post_df['comment_count'], post_df['sentiment_score'],
                                 c=post_df['sentiment_score'], cmap='RdYlGn',
                                 s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

            # Color code by post type
            giveaway_posts = post_df[post_df['is_giveaway']]
            pr_posts = post_df[post_df['is_pr_recruitment']]

            if len(giveaway_posts) > 0:
                ax2.scatter(giveaway_posts['comment_count'], giveaway_posts['sentiment_score'],
                           marker='s', s=120, alpha=0.8, edgecolors='blue', linewidth=2,
                           facecolors='none', label='Giveaway Posts')

            if len(pr_posts) > 0:
                ax2.scatter(pr_posts['comment_count'], pr_posts['sentiment_score'],
                           marker='^', s=120, alpha=0.8, edgecolors='purple', linewidth=2,
                           facecolors='none', label='PR Recruitment')

            ax2.set_xlabel('Number of Comments Analyzed')
            ax2.set_ylabel('Sentiment Score (Positive % - Negative %)')
            ax2.set_title('Post Performance: Sentiment Score vs Engagement', fontweight='bold', fontsize=14, pad=20)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Sentiment Score', rotation=270, labelpad=20)

            if len(giveaway_posts) > 0 or len(pr_posts) > 0:
                ax2.legend()

            plt.tight_layout()
            post_sentiment_path = os.path.join(viz_dir, 'post_sentiment_scores.png')
            plt.savefig(post_sentiment_path, dpi=300, bbox_inches='tight')
            print(f"📈 Post sentiment scores chart saved as '{post_sentiment_path}'")
            plt.close()

            # 3. Top and bottom performing posts
            fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(14, 10))

            # Top 5 posts by sentiment score
            top_posts = post_df.head(5)
            if len(top_posts) > 0:
                bars_top = ax3a.barh(range(len(top_posts)), top_posts['sentiment_score'],
                                    color='lightgreen', alpha=0.8)
                ax3a.set_yticks(range(len(top_posts)))
                ax3a.set_yticklabels([f"Post {i+1}: {caption[:40]}..."
                                     for i, caption in enumerate(top_posts['post_caption_preview'])])
                ax3a.set_xlabel('Sentiment Score')
                ax3a.set_title('Top 5 Posts by Sentiment Score', fontweight='bold')
                ax3a.grid(True, alpha=0.3, axis='x')

                # Add score labels
                for i, (bar, score) in enumerate(zip(bars_top, top_posts['sentiment_score'])):
                    ax3a.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                             f'{score:.1f}%', va='center', fontweight='bold')

            # Bottom 5 posts by sentiment score (if any negative)
            bottom_posts = post_df.tail(5)
            if len(bottom_posts) > 0 and bottom_posts['sentiment_score'].min() < 0:
                bars_bottom = ax3b.barh(range(len(bottom_posts)), bottom_posts['sentiment_score'],
                                       color='lightcoral', alpha=0.8)
                ax3b.set_yticks(range(len(bottom_posts)))
                ax3b.set_yticklabels([f"Post {i+1}: {caption[:40]}..."
                                     for i, caption in enumerate(bottom_posts['post_caption_preview'])])
                ax3b.set_xlabel('Sentiment Score')
                ax3b.set_title('Posts Needing Attention (Negative Sentiment)', fontweight='bold')
                ax3b.grid(True, alpha=0.3, axis='x')

                # Add score labels
                for i, (bar, score) in enumerate(zip(bars_bottom, bottom_posts['sentiment_score'])):
                    ax3b.text(bar.get_width() - 1, bar.get_y() + bar.get_height()/2,
                             f'{score:.1f}%', va='center', fontweight='bold')
            else:
                ax3b.text(0.5, 0.5, 'No posts with negative sentiment detected',
                         ha='center', va='center', transform=ax3b.transAxes,
                         fontsize=12, style='italic')
                ax3b.set_xlim(0, 1)
                ax3b.set_ylim(0, 1)
                ax3b.set_title('Posts Needing Attention (None Found)', fontweight='bold')

            plt.tight_layout()
            top_bottom_path = os.path.join(viz_dir, 'top_bottom_posts_sentiment.png')
            plt.savefig(top_bottom_path, dpi=300, bbox_inches='tight')
            print(f"🏆 Top/bottom posts chart saved as '{top_bottom_path}'")
            plt.close()

        # 4. Sentiment by post type comparison
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(15, 6))

        post_type_analysis = analysis_results['post_type_analysis']

        # Giveaway posts
        giveaway_data = post_type_analysis['giveaway_posts']['sentiment_distribution']
        if giveaway_data:
            ax4a.pie(giveaway_data.values(), labels=giveaway_data.keys(), autopct='%1.1f%%',
                    colors=[colors.get(k, 'lightblue') for k in giveaway_data.keys()])
            ax4a.set_title(f'Giveaway Posts Sentiment\n({post_type_analysis["giveaway_posts"]["count"]} comments)',
                          fontweight='bold')

        # Regular posts
        regular_data = post_type_analysis['regular_posts']['sentiment_distribution']
        if regular_data:
            ax4b.pie(regular_data.values(), labels=regular_data.keys(), autopct='%1.1f%%',
                    colors=[colors.get(k, 'lightblue') for k in regular_data.keys()])
            ax4b.set_title(f'Regular Posts Sentiment\n({post_type_analysis["regular_posts"]["count"]} comments)',
                          fontweight='bold')

        plt.tight_layout()
        post_type_path = os.path.join(viz_dir, 'sentiment_by_post_type.png')
        plt.savefig(post_type_path, dpi=300, bbox_inches='tight')
        print(f"🎁 Post type sentiment chart saved as '{post_type_path}'")
        plt.close()

        return True
    
    def generate_reputation_report(self, sentiment_df: pd.DataFrame, analysis_results: Dict) -> str:
        """Generate a comprehensive reputation report"""
        total_comments = len(sentiment_df)
        sentiment_counts = sentiment_df['sentiment'].value_counts()
        
        positive_pct = (sentiment_counts.get('positive', 0) / total_comments) * 100
        negative_pct = (sentiment_counts.get('negative', 0) / total_comments) * 100
        neutral_pct = (sentiment_counts.get('neutral', 0) / total_comments) * 100
        
        report = f"""
# TreeHut Brand Reputation Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- **Total Comments Analyzed**: {total_comments:,}
- **Overall Sentiment**: {positive_pct:.1f}% Positive, {negative_pct:.1f}% Negative, {neutral_pct:.1f}% Neutral
- **Brand Health Score**: {positive_pct - negative_pct:.1f}/100

## Key Findings

### Sentiment Distribution
- **Positive**: {sentiment_counts.get('positive', 0)} comments ({positive_pct:.1f}%)
- **Negative**: {sentiment_counts.get('negative', 0)} comments ({negative_pct:.1f}%)
- **Neutral**: {sentiment_counts.get('neutral', 0)} comments ({neutral_pct:.1f}%)

### Top Themes Overall
"""
        
        themes = analysis_results['themes']['overall_themes']
        for theme, count in list(themes.items())[:5]:
            report += f"- **{theme.replace('_', ' ').title()}**: {count} mentions\n"
        
        report += "\n### Positive Feedback Themes\n"
        pos_themes = analysis_results['themes']['themes_by_sentiment'].get('positive', {})
        for theme, count in list(pos_themes.items())[:3]:
            report += f"- **{theme.replace('_', ' ').title()}**: {count} mentions\n"
        
        if analysis_results['themes']['themes_by_sentiment'].get('negative'):
            report += "\n### Areas for Improvement\n"
            neg_themes = analysis_results['themes']['themes_by_sentiment'].get('negative', {})
            for theme, count in list(neg_themes.items())[:3]:
                report += f"- **{theme.replace('_', ' ').title()}**: {count} mentions\n"
        
        return report

if __name__ == "__main__":
    import sys
    
    # Check for API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("❌ Please set your Anthropic API key:")
        print("   export ANTHROPIC_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = TreeHutSentimentAnalyzer()
    
    # Get sample size from command line or use default
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    
    print(f"🚀 Starting sentiment analysis with sample size: {sample_size}")
    print("⚠️  Note: This will make API calls to Claude - costs may apply")
    
    # Run sentiment analysis
    sentiment_results = analyzer.analyze_sample_comments(sample_size=sample_size)

    # Analyze results
    post_analysis = analyzer.analyze_by_individual_posts(sentiment_results)
    post_type_analysis = analyzer.analyze_by_post_type(sentiment_results)
    theme_analysis = analyzer.extract_themes(sentiment_results)

    # Combine results
    analysis_results = {
        'post_analysis': post_analysis,
        'post_type_analysis': post_type_analysis,
        'themes': theme_analysis
    }

    # Save detailed post analysis
    post_analysis.to_csv('sentiment_analysis/post_sentiment_analysis.csv', index=False)
    print(f"📊 Detailed post analysis saved to: post_sentiment_analysis.csv")
    
    # Create visualizations
    analyzer.create_sentiment_visualizations(sentiment_results, analysis_results)
    
    # Generate and save report
    report = analyzer.generate_reputation_report(sentiment_results, analysis_results)
    
    with open('sentiment_analysis/brand_reputation_report.md', 'w') as f:
        f.write(report)
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Visualizations saved to: visualizations/brand_reputation/")
    print(f"📄 Report saved to: brand_reputation_report.md")
    print(f"\n💡 Usage: python sentiment_analysis.py [sample_size]")
    print(f"   Default sample size: 50 comments")
