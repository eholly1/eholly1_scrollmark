#!/usr/bin/env python3
"""
@treehut Instagram Engagement Analysis
Comprehensive analysis of March 2025 engagement data for strategic insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TreeHutAnalyzer:
    def __init__(self, csv_path='engagements.csv'):
        """Initialize the analyzer with engagement data"""
        print("Loading engagement data...")
        self.df = pd.read_csv(csv_path)
        self.prepare_data()
        
    def prepare_data(self):
        """Clean and prepare the data for analysis"""
        print(f"Loaded {len(self.df):,} engagement records")
        
        # Convert timestamp to datetime with flexible format
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], format='mixed')
        self.df['date'] = self.df['timestamp'].dt.date
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.day_name()
        
        # Clean text data
        self.df['comment_text'] = self.df['comment_text'].fillna('')
        self.df['media_caption'] = self.df['media_caption'].fillna('')
        
        # Calculate comment length
        self.df['comment_length'] = self.df['comment_text'].str.len()
        
        print("Data preparation complete!")
        
    def data_overview(self):
        """Generate comprehensive data overview"""
        print("\n" + "="*60)
        print("📊 DATA OVERVIEW & QUALITY ASSESSMENT")
        print("="*60)
        
        # Basic statistics
        print(f"\n📈 Dataset Statistics:")
        print(f"• Total Comments: {len(self.df):,}")
        print(f"• Date Range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"• Unique Posts: {self.df['media_id'].nunique():,}")
        print(f"• Average Comments per Post: {len(self.df) / self.df['media_id'].nunique():.1f}")
        
        # Engagement distribution by post
        post_engagement = self.df.groupby('media_id').size().sort_values(ascending=False)
        print(f"\n🔥 Top Performing Posts (by comment count):")
        for i, (media_id, count) in enumerate(post_engagement.head(5).items()):
            caption = self.df[self.df['media_id'] == media_id]['media_caption'].iloc[0]
            caption_preview = caption[:80] + "..." if len(caption) > 80 else caption
            print(f"  {i+1}. {count:,} comments - {caption_preview}")
            
        # Daily engagement patterns
        daily_engagement = self.df.groupby('date').size()
        print(f"\n📅 Daily Engagement Patterns:")
        print(f"• Peak Day: {daily_engagement.idxmax()} ({daily_engagement.max():,} comments)")
        print(f"• Lowest Day: {daily_engagement.idxmin()} ({daily_engagement.min():,} comments)")
        print(f"• Average Daily Comments: {daily_engagement.mean():.0f}")
        
        # Hourly patterns
        hourly_engagement = self.df.groupby('hour').size()
        peak_hours = hourly_engagement.nlargest(3)
        print(f"\n⏰ Peak Engagement Hours:")
        for hour, count in peak_hours.items():
            print(f"  • {hour:02d}:00 - {count:,} comments")
            
        return {
            'total_comments': len(self.df),
            'unique_posts': self.df['media_id'].nunique(),
            'date_range': (self.df['date'].min(), self.df['date'].max()),
            'top_posts': post_engagement.head(10),
            'daily_engagement': daily_engagement,
            'hourly_engagement': hourly_engagement
        }
    
    def content_analysis(self):
        """Analyze content themes and performance"""
        print("\n" + "="*60)
        print("📝 CONTENT PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Analyze post captions for themes
        all_captions = self.df['media_caption'].unique()
        
        # Product mentions in captions
        product_keywords = {
            'scrub': ['scrub', 'exfoliat'],
            'lotion': ['lotion', 'moisturiz'],
            'hand_wash': ['hand wash', 'handwash'],
            'shave': ['shave', 'pre-shave'],
            'serum': ['serum'],
            'oil': ['oil']
        }
        
        scent_keywords = {
            'vanilla': ['vanilla'],
            'tangerine': ['tangerine', 'orange'],
            'coconut': ['coconut'],
            'shea': ['shea'],
            'tropical': ['tropical', 'mango', 'pineapple'],
            'berry': ['berry', 'strawberry', 'raspberry'],
            'citrus': ['citrus', 'lemon', 'lime']
        }
        
        # Analyze product mentions
        product_performance = {}
        for product, keywords in product_keywords.items():
            pattern = '|'.join(keywords)
            matching_posts = self.df[self.df['media_caption'].str.contains(pattern, case=False, na=False)]
            if len(matching_posts) > 0:
                avg_engagement = len(matching_posts) / matching_posts['media_id'].nunique()
                product_performance[product] = {
                    'posts': matching_posts['media_id'].nunique(),
                    'total_comments': len(matching_posts),
                    'avg_comments_per_post': avg_engagement
                }
        
        print(f"\n🛍️ Product Performance:")
        for product, stats in sorted(product_performance.items(), 
                                   key=lambda x: x[1]['avg_comments_per_post'], reverse=True):
            print(f"  • {product.title()}: {stats['posts']} posts, "
                  f"{stats['total_comments']:,} comments, "
                  f"{stats['avg_comments_per_post']:.1f} avg/post")
        
        # Analyze scent mentions
        scent_performance = {}
        for scent, keywords in scent_keywords.items():
            pattern = '|'.join(keywords)
            matching_posts = self.df[self.df['media_caption'].str.contains(pattern, case=False, na=False)]
            if len(matching_posts) > 0:
                avg_engagement = len(matching_posts) / matching_posts['media_id'].nunique()
                scent_performance[scent] = {
                    'posts': matching_posts['media_id'].nunique(),
                    'total_comments': len(matching_posts),
                    'avg_comments_per_post': avg_engagement
                }
        
        print(f"\n🌸 Scent Performance:")
        for scent, stats in sorted(scent_performance.items(), 
                                 key=lambda x: x[1]['avg_comments_per_post'], reverse=True):
            print(f"  • {scent.title()}: {stats['posts']} posts, "
                  f"{stats['total_comments']:,} comments, "
                  f"{stats['avg_comments_per_post']:.1f} avg/post")
        
        # Analyze giveaway performance
        giveaway_posts = self.df[self.df['media_caption'].str.contains('giveaway|contest|win', case=False, na=False)]
        if len(giveaway_posts) > 0:
            giveaway_engagement = len(giveaway_posts) / giveaway_posts['media_id'].nunique()
            print(f"\n🎁 Giveaway Performance:")
            print(f"  • Giveaway Posts: {giveaway_posts['media_id'].nunique()}")
            print(f"  • Total Comments: {len(giveaway_posts):,}")
            print(f"  • Avg Comments per Giveaway: {giveaway_engagement:.1f}")
        
        return {
            'product_performance': product_performance,
            'scent_performance': scent_performance,
            'giveaway_stats': len(giveaway_posts) if len(giveaway_posts) > 0 else 0
        }

    def create_visualizations(self):
        """Create key visualization plots"""
        print("\n" + "="*60)
        print("📊 CREATING VISUALIZATIONS")
        print("="*60)

        # Create visualizations directory structure
        viz_dir = 'visualizations'
        core_dir = os.path.join(viz_dir, 'core_engagement')
        product_dir = os.path.join(viz_dir, 'product_scent_analysis')

        for directory in [viz_dir, core_dir, product_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"📁 Created directory: {directory}/")

        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Daily engagement pattern
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        daily_engagement = self.df.groupby('date').size()
        ax1.plot(daily_engagement.index, daily_engagement.values, marker='o', linewidth=2, markersize=4, color='steelblue')
        ax1.set_title('Daily Engagement Pattern', fontweight='bold', fontsize=14, pad=20)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Number of Comments')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        daily_path = os.path.join(core_dir, 'daily_engagement_pattern.png')
        plt.savefig(daily_path, dpi=300, bbox_inches='tight')
        print(f"📅 Daily engagement pattern saved as '{daily_path}'")
        plt.close()

        # 2. Hourly engagement distribution
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        hourly_engagement = self.df.groupby('hour').size()
        ax2.bar(hourly_engagement.index, hourly_engagement.values, color='skyblue', alpha=0.7)
        ax2.set_title('Hourly Engagement Distribution', fontweight='bold', fontsize=14, pad=20)
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Number of Comments')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        hourly_path = os.path.join(core_dir, 'hourly_engagement_distribution.png')
        plt.savefig(hourly_path, dpi=300, bbox_inches='tight')
        print(f"⏰ Hourly engagement distribution saved as '{hourly_path}'")
        plt.close()

        # 3. Top posts by engagement
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        post_engagement = self.df.groupby('media_id').size().sort_values(ascending=False).head(10)
        post_labels = [f"Post {i+1}" for i in range(len(post_engagement))]
        ax3.barh(post_labels, post_engagement.values, color='lightcoral', alpha=0.7)
        ax3.set_title('Top 10 Posts by Comment Count', fontweight='bold', fontsize=14, pad=20)
        ax3.set_xlabel('Number of Comments')
        plt.tight_layout()
        top_posts_path = os.path.join(core_dir, 'top_posts_by_comments.png')
        plt.savefig(top_posts_path, dpi=300, bbox_inches='tight')
        print(f"🔥 Top posts chart saved as '{top_posts_path}'")
        plt.close()

        # 4. Giveaway vs Regular post performance
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        giveaway_posts = self.df[self.df['media_caption'].str.contains('giveaway|contest|win', case=False, na=False)]
        regular_posts = self.df[~self.df['media_caption'].str.contains('giveaway|contest|win', case=False, na=False)]

        giveaway_avg = len(giveaway_posts) / giveaway_posts['media_id'].nunique() if len(giveaway_posts) > 0 else 0
        regular_avg = len(regular_posts) / regular_posts['media_id'].nunique() if len(regular_posts) > 0 else 0

        post_types = ['Regular Posts', 'Giveaway Posts']
        avg_engagement = [regular_avg, giveaway_avg]
        colors = ['lightblue', 'gold']

        bars = ax4.bar(post_types, avg_engagement, color=colors, alpha=0.7)
        ax4.set_title('Average Engagement: Regular vs Giveaway Posts', fontweight='bold', fontsize=14, pad=20)
        ax4.set_ylabel('Average Comments per Post')

        # Add value labels on bars
        for bar, value in zip(bars, avg_engagement):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        giveaway_comparison_path = os.path.join(core_dir, 'giveaway_vs_regular_posts.png')
        plt.savefig(giveaway_comparison_path, dpi=300, bbox_inches='tight')
        print(f"🎁 Giveaway comparison chart saved as '{giveaway_comparison_path}'")
        plt.close()

        # 5. Product performance analysis
        fig5, ax5 = plt.subplots(figsize=(12, 8))
        product_keywords = {
            'scrub': ['scrub', 'exfoliat'],
            'lotion': ['lotion', 'moisturiz'],
            'hand_wash': ['hand wash', 'handwash'],
            'shave': ['shave', 'pre-shave'],
            'serum': ['serum'],
            'oil': ['oil']
        }

        product_performance = {}
        for product, keywords in product_keywords.items():
            pattern = '|'.join(keywords)
            matching_posts = self.df[self.df['media_caption'].str.contains(pattern, case=False, na=False)]
            if len(matching_posts) > 0:
                avg_engagement = len(matching_posts) / matching_posts['media_id'].nunique()
                product_performance[product] = avg_engagement

        if product_performance:
            products = list(product_performance.keys())
            performance = list(product_performance.values())

            ax5.barh(products, performance, color='lightgreen', alpha=0.7)
            ax5.set_title('Average Engagement by Product Type', fontweight='bold', fontsize=14, pad=20)
            ax5.set_xlabel('Average Comments per Post')
            ax5.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        product_performance_path = os.path.join(product_dir, 'product_performance_analysis.png')
        plt.savefig(product_performance_path, dpi=300, bbox_inches='tight')
        print(f"🛍️ Product performance analysis saved as '{product_performance_path}'")
        plt.close()

        # 6. Scent performance analysis
        fig6, ax6 = plt.subplots(figsize=(12, 8))
        scent_keywords = {
            'vanilla': ['vanilla'],
            'tangerine': ['tangerine', 'orange'],
            'coconut': ['coconut'],
            'shea': ['shea'],
            'tropical': ['tropical', 'mango', 'pineapple'],
            'berry': ['berry', 'strawberry', 'raspberry'],
            'citrus': ['citrus', 'lemon', 'lime']
        }

        scent_performance = {}
        for scent, keywords in scent_keywords.items():
            pattern = '|'.join(keywords)
            matching_posts = self.df[self.df['media_caption'].str.contains(pattern, case=False, na=False)]
            if len(matching_posts) > 0:
                avg_engagement = len(matching_posts) / matching_posts['media_id'].nunique()
                scent_performance[scent] = avg_engagement

        if scent_performance:
            scents = list(scent_performance.keys())
            performance = list(scent_performance.values())

            ax6.barh(scents, performance, color='lightpink', alpha=0.7)
            ax6.set_title('Average Engagement by Scent Type', fontweight='bold', fontsize=14, pad=20)
            ax6.set_xlabel('Average Comments per Post')
            ax6.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        scent_performance_path = os.path.join(product_dir, 'scent_performance_analysis.png')
        plt.savefig(scent_performance_path, dpi=300, bbox_inches='tight')
        print(f"🌸 Scent performance analysis saved as '{scent_performance_path}'")
        plt.close()

        return True

    def create_additional_visualizations(self):
        """Create additional specialized charts for customer insights"""
        print("\n" + "="*60)
        print("📊 CREATING ADDITIONAL CUSTOMER INSIGHT VISUALIZATIONS")
        print("="*60)

        # Create visualizations directory structure
        viz_dir = 'visualizations'
        insights_dir = os.path.join(viz_dir, 'customer_insights')

        for directory in [viz_dir, insights_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # 1. Scent Performance Comparison Chart
        self._create_scent_performance_chart(insights_dir)

        # 2. Geographic Demand Analysis
        self._create_geographic_demand_chart(insights_dir)

        # 3. Product Category Trend Over Time
        self._create_product_trend_chart(insights_dir)

        # 4. Comment Sentiment by Product Type
        self._create_sentiment_analysis_chart(insights_dir)

        # 5. Engagement vs Post Frequency Scatter Plot
        self._create_engagement_frequency_scatter(insights_dir)

        print("✅ All additional visualizations created successfully!")
        return True

    def _create_scent_performance_chart(self, viz_dir):
        """Create scent performance comparison with sample sizes"""
        scent_keywords = {
            'vanilla': ['vanilla'],
            'tangerine': ['tangerine', 'orange'],
            'coconut': ['coconut'],
            'shea': ['shea'],
            'tropical': ['tropical', 'mango', 'pineapple'],
            'berry': ['berry', 'strawberry', 'raspberry'],
            'citrus': ['citrus', 'lemon', 'lime']
        }

        scent_data = []
        for scent, keywords in scent_keywords.items():
            pattern = '|'.join(keywords)
            matching_posts = self.df[self.df['media_caption'].str.contains(pattern, case=False, na=False)]
            if len(matching_posts) > 0:
                unique_posts = matching_posts['media_id'].nunique()
                total_comments = len(matching_posts)
                avg_engagement = total_comments / unique_posts
                scent_data.append({
                    'scent': scent.title(),
                    'avg_engagement': avg_engagement,
                    'post_count': unique_posts,
                    'total_comments': total_comments
                })

        if scent_data:
            scent_df = pd.DataFrame(scent_data).sort_values('avg_engagement', ascending=True)

            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(scent_df['scent'], scent_df['avg_engagement'],
                          color='lightcoral', alpha=0.7)

            # Add sample size annotations
            for i, (bar, row) in enumerate(zip(bars, scent_df.itertuples())):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                       f'{row.post_count} posts\n{row.total_comments} comments',
                       ha='left', va='center', fontsize=9, alpha=0.8)

            ax.set_title('Scent Performance: Engagement Rate with Sample Sizes',
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Average Comments per Post')
            ax.set_ylabel('Scent Category')
            ax.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            scent_path = os.path.join(viz_dir, 'scent_performance_comparison.png')
            plt.savefig(scent_path, dpi=300, bbox_inches='tight')
            print(f"📈 Scent performance chart saved as '{scent_path}'")
            plt.close()

    def _create_geographic_demand_chart(self, viz_dir):
        """Create geographic demand analysis from comments"""
        # Common location keywords to search for
        locations = {
            'Canada': ['canada', 'canadian'],
            'UK': ['uk', 'britain', 'england', 'scotland', 'wales'],
            'Australia': ['australia', 'aussie', 'oz'],
            'Europe': ['europe', 'european'],
            'Mexico': ['mexico', 'mexican'],
            'International': ['international', 'worldwide', 'global']
        }

        location_mentions = {}
        for location, keywords in locations.items():
            pattern = '|'.join(keywords)
            mentions = self.df[self.df['comment_text'].str.contains(pattern, case=False, na=False)]
            location_mentions[location] = len(mentions)

        # Filter out locations with no mentions
        location_mentions = {k: v for k, v in location_mentions.items() if v > 0}

        if location_mentions:
            locations_list = list(location_mentions.keys())
            counts = list(location_mentions.values())

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(locations_list, counts, color='skyblue', alpha=0.7)

            # Add count labels on bars
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       str(count), ha='center', va='bottom', fontweight='bold')

            ax.set_title('Geographic Demand Signals in Customer Comments',
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Location/Region')
            ax.set_ylabel('Number of Mentions in Comments')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45)

            plt.tight_layout()
            geo_path = os.path.join(viz_dir, 'geographic_demand_analysis.png')
            plt.savefig(geo_path, dpi=300, bbox_inches='tight')
            print(f"🌍 Geographic demand chart saved as '{geo_path}'")
            plt.close()

    def _create_product_trend_chart(self, viz_dir):
        """Create product category engagement trends over time"""
        product_keywords = {
            'scrub': ['scrub', 'exfoliat'],
            'lotion': ['lotion', 'moisturiz'],
            'hand_wash': ['hand wash', 'handwash'],
            'shave': ['shave', 'pre-shave'],
            'serum': ['serum'],
            'oil': ['oil']
        }

        # Group by week for trend analysis
        self.df['week'] = self.df['timestamp'].dt.to_period('W')

        trend_data = {}
        for product, keywords in product_keywords.items():
            pattern = '|'.join(keywords)
            product_posts = self.df[self.df['media_caption'].str.contains(pattern, case=False, na=False)]
            weekly_engagement = product_posts.groupby('week').size()
            trend_data[product.title()] = weekly_engagement

        if trend_data:
            fig, ax = plt.subplots(figsize=(12, 8))

            for product, weekly_data in trend_data.items():
                if len(weekly_data) > 1:  # Only plot if we have multiple data points
                    ax.plot(weekly_data.index.astype(str), weekly_data.values,
                           marker='o', linewidth=2, label=product, alpha=0.8)

            ax.set_title('Product Category Engagement Trends Over Time',
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Week')
            ax.set_ylabel('Number of Comments')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

            plt.tight_layout()
            trend_path = os.path.join(viz_dir, 'product_category_trends.png')
            plt.savefig(trend_path, dpi=300, bbox_inches='tight')
            print(f"📈 Product trend chart saved as '{trend_path}'")
            plt.close()

    def _create_sentiment_analysis_chart(self, viz_dir):
        """Create basic sentiment analysis by product type"""
        # Simple sentiment keywords (basic approach)
        positive_words = ['love', 'amazing', 'great', 'awesome', 'perfect', 'best', 'good', 'nice', 'beautiful']
        negative_words = ['hate', 'bad', 'terrible', 'awful', 'worst', 'disappointed', 'sucks']

        product_keywords = {
            'scrub': ['scrub', 'exfoliat'],
            'lotion': ['lotion', 'moisturiz'],
            'hand_wash': ['hand wash', 'handwash'],
            'shave': ['shave', 'pre-shave'],
            'serum': ['serum'],
            'oil': ['oil']
        }

        sentiment_data = []
        for product, keywords in product_keywords.items():
            pattern = '|'.join(keywords)
            product_comments = self.df[self.df['media_caption'].str.contains(pattern, case=False, na=False)]

            if len(product_comments) > 0:
                positive_count = 0
                negative_count = 0
                total_comments = len(product_comments)

                for comment in product_comments['comment_text']:
                    comment_lower = str(comment).lower()
                    if any(word in comment_lower for word in positive_words):
                        positive_count += 1
                    elif any(word in comment_lower for word in negative_words):
                        negative_count += 1

                neutral_count = total_comments - positive_count - negative_count

                sentiment_data.append({
                    'product': product.title(),
                    'positive': positive_count,
                    'negative': negative_count,
                    'neutral': neutral_count,
                    'total': total_comments
                })

        if sentiment_data:
            sentiment_df = pd.DataFrame(sentiment_data)

            fig, ax = plt.subplots(figsize=(12, 8))

            # Create stacked bar chart
            products = sentiment_df['product']
            positive_pct = (sentiment_df['positive'] / sentiment_df['total']) * 100
            negative_pct = (sentiment_df['negative'] / sentiment_df['total']) * 100
            neutral_pct = (sentiment_df['neutral'] / sentiment_df['total']) * 100

            ax.bar(products, positive_pct, label='Positive', color='lightgreen', alpha=0.8)
            ax.bar(products, negative_pct, bottom=positive_pct, label='Negative', color='lightcoral', alpha=0.8)
            ax.bar(products, neutral_pct, bottom=positive_pct + negative_pct, label='Neutral', color='lightgray', alpha=0.8)

            ax.set_title('Comment Sentiment Distribution by Product Type',
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Product Category')
            ax.set_ylabel('Percentage of Comments')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45)

            plt.tight_layout()
            sentiment_path = os.path.join(viz_dir, 'sentiment_by_product.png')
            plt.savefig(sentiment_path, dpi=300, bbox_inches='tight')
            print(f"😊 Sentiment analysis chart saved as '{sentiment_path}'")
            plt.close()

    def _create_engagement_frequency_scatter(self, viz_dir):
        """Create scatter plot of engagement vs post frequency"""
        product_keywords = {
            'scrub': ['scrub', 'exfoliat'],
            'lotion': ['lotion', 'moisturiz'],
            'hand_wash': ['hand wash', 'handwash'],
            'shave': ['shave', 'pre-shave'],
            'serum': ['serum'],
            'oil': ['oil']
        }

        scent_keywords = {
            'vanilla': ['vanilla'],
            'tangerine': ['tangerine', 'orange'],
            'coconut': ['coconut'],
            'shea': ['shea'],
            'tropical': ['tropical', 'mango', 'pineapple'],
            'berry': ['berry', 'strawberry', 'raspberry'],
            'citrus': ['citrus', 'lemon', 'lime']
        }

        # Combine both product and scent data
        all_categories = {**product_keywords, **scent_keywords}

        scatter_data = []
        for category, keywords in all_categories.items():
            pattern = '|'.join(keywords)
            matching_posts = self.df[self.df['media_caption'].str.contains(pattern, case=False, na=False)]
            if len(matching_posts) > 0:
                post_count = matching_posts['media_id'].nunique()
                avg_engagement = len(matching_posts) / post_count
                category_type = 'Product' if category in product_keywords else 'Scent'

                scatter_data.append({
                    'category': category.title(),
                    'post_count': post_count,
                    'avg_engagement': avg_engagement,
                    'type': category_type
                })

        if scatter_data:
            scatter_df = pd.DataFrame(scatter_data)

            fig, ax = plt.subplots(figsize=(12, 8))

            # Different colors for products vs scents
            for cat_type in scatter_df['type'].unique():
                subset = scatter_df[scatter_df['type'] == cat_type]
                ax.scatter(subset['post_count'], subset['avg_engagement'],
                          label=cat_type, alpha=0.7, s=100)

                # Add labels for each point
                for _, row in subset.iterrows():
                    ax.annotate(row['category'],
                               (row['post_count'], row['avg_engagement']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=9, alpha=0.8)

            ax.set_title('Engagement vs Post Frequency: Identifying Over/Under-Exploited Categories',
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Number of Posts')
            ax.set_ylabel('Average Comments per Post')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            scatter_path = os.path.join(viz_dir, 'engagement_vs_frequency_scatter.png')
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            print(f"📊 Engagement vs frequency scatter plot saved as '{scatter_path}'")
            plt.close()

if __name__ == "__main__":
    import sys

    # Initialize analyzer
    analyzer = TreeHutAnalyzer()

    # Check if user wants visualizations
    if len(sys.argv) > 1 and sys.argv[1] == '--plots':
        # Run analysis with all visualizations
        overview_results = analyzer.data_overview()
        content_results = analyzer.content_analysis()
        analyzer.create_visualizations()
        analyzer.create_additional_visualizations()
    else:
        # Run text-only analysis
        overview_results = analyzer.data_overview()
        content_results = analyzer.content_analysis()

    print("\n" + "="*60)
    print("✅ INITIAL ANALYSIS COMPLETE")
    print("="*60)
    print("💡 Usage options:")
    print("   python treehut_analysis.py          # Text analysis only")
    print("   python treehut_analysis.py --plots  # All visualizations (11 charts)")
    print("Next steps: Run sentiment analysis, community behavior analysis, and strategic recommendations")
