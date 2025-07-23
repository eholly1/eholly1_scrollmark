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

        # Create visualizations directory if it doesn't exist
        viz_dir = 'visualizations'
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
            print(f"📁 Created directory: {viz_dir}/")

        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('@treehut Instagram Engagement Analysis - March 2025', fontsize=16, fontweight='bold')

        # 1. Daily engagement pattern
        daily_engagement = self.df.groupby('date').size()
        axes[0,0].plot(daily_engagement.index, daily_engagement.values, marker='o', linewidth=2, markersize=4)
        axes[0,0].set_title('Daily Engagement Pattern', fontweight='bold')
        axes[0,0].set_xlabel('Date')
        axes[0,0].set_ylabel('Number of Comments')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)

        # 2. Hourly engagement heatmap
        hourly_engagement = self.df.groupby('hour').size()
        axes[0,1].bar(hourly_engagement.index, hourly_engagement.values, color='skyblue', alpha=0.7)
        axes[0,1].set_title('Hourly Engagement Distribution', fontweight='bold')
        axes[0,1].set_xlabel('Hour of Day')
        axes[0,1].set_ylabel('Number of Comments')
        axes[0,1].grid(True, alpha=0.3)

        # 3. Top posts by engagement
        post_engagement = self.df.groupby('media_id').size().sort_values(ascending=False).head(10)
        post_labels = [f"Post {i+1}" for i in range(len(post_engagement))]
        axes[1,0].barh(post_labels, post_engagement.values, color='lightcoral', alpha=0.7)
        axes[1,0].set_title('Top 10 Posts by Comment Count', fontweight='bold')
        axes[1,0].set_xlabel('Number of Comments')

        # 4. Giveaway vs Regular post performance
        giveaway_posts = self.df[self.df['media_caption'].str.contains('giveaway|contest|win', case=False, na=False)]
        regular_posts = self.df[~self.df['media_caption'].str.contains('giveaway|contest|win', case=False, na=False)]

        giveaway_avg = len(giveaway_posts) / giveaway_posts['media_id'].nunique() if len(giveaway_posts) > 0 else 0
        regular_avg = len(regular_posts) / regular_posts['media_id'].nunique() if len(regular_posts) > 0 else 0

        post_types = ['Regular Posts', 'Giveaway Posts']
        avg_engagement = [regular_avg, giveaway_avg]
        colors = ['lightblue', 'gold']

        bars = axes[1,1].bar(post_types, avg_engagement, color=colors, alpha=0.7)
        axes[1,1].set_title('Average Engagement: Regular vs Giveaway Posts', fontweight='bold')
        axes[1,1].set_ylabel('Average Comments per Post')

        # Add value labels on bars
        for bar, value in zip(bars, avg_engagement):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                          f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        engagement_plot_path = os.path.join(viz_dir, 'treehut_engagement_analysis.png')
        plt.savefig(engagement_plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Engagement analysis saved as '{engagement_plot_path}'")
        plt.show()

        # Create a second figure for product/scent analysis
        fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
        fig2.suptitle('Product & Scent Performance Analysis', fontsize=16, fontweight='bold')

        # Product performance
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

            axes2[0].barh(products, performance, color='lightgreen', alpha=0.7)
            axes2[0].set_title('Average Engagement by Product Type', fontweight='bold')
            axes2[0].set_xlabel('Average Comments per Post')

        # Scent performance
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

            axes2[1].barh(scents, performance, color='lightpink', alpha=0.7)
            axes2[1].set_title('Average Engagement by Scent Type', fontweight='bold')
            axes2[1].set_xlabel('Average Comments per Post')

        plt.tight_layout()
        product_plot_path = os.path.join(viz_dir, 'treehut_product_scent_analysis.png')
        plt.savefig(product_plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Product/Scent analysis saved as '{product_plot_path}'")
        plt.show()

        return True

if __name__ == "__main__":
    import sys

    # Initialize analyzer
    analyzer = TreeHutAnalyzer()

    # Check if user wants visualizations
    if len(sys.argv) > 1 and sys.argv[1] == '--plots':
        # Run analysis with visualizations
        overview_results = analyzer.data_overview()
        content_results = analyzer.content_analysis()
        analyzer.create_visualizations()
    else:
        # Run text-only analysis
        overview_results = analyzer.data_overview()
        content_results = analyzer.content_analysis()

    print("\n" + "="*60)
    print("✅ INITIAL ANALYSIS COMPLETE")
    print("="*60)
    print("💡 To see visualizations, run: python treehut_analysis.py --plots")
    print("Next steps: Run sentiment analysis, community behavior analysis, and strategic recommendations")
