# @treehut Social Media Analytics Project

## Project Goals

This project analyzes Instagram engagement data for @treehut to provide actionable insights across five key social media management objectives:

• **1. Brand Awareness & Reach** - Grow follower base and expand brand visibility in the beauty/skincare space

• **2. Customer Insights & Feedback** - Gather product feedback and identify market expansion opportunities

• **3. Sales & Conversion** - Drive traffic and convert social engagement into purchases

• **4. Brand Reputation Management** - Monitor sentiment and maintain consistent brand voice

## Approach

Using the `engagements.csv` dataset containing Instagram post metadata and user comments from March 2025, we will:

- **Define metrics & visualize** to measure performance and track progress
- **Provide insights** about customer behavior and preferences for each goal area
- **Generate actionable feedback** for social media strategy optimization

The analysis bridges marketing intent with customer reality, enabling data-driven decisions that move business metrics beyond vanity engagement numbers.

## Analysis Results

For detailed findings and strategic recommendations across all five business objectives, see:

**[📊 Final Insights & Recommendations](insights_and_recommendations.md)**

## Usage

### Engagement Analysis & Visualizations

```bash
# Text analysis only
python treehut_analysis.py

# All visualizations (11 charts in organized subdirectories)
python treehut_analysis.py --plots
```

**Outputs:**
- Text summary of engagement patterns, product performance, and content insights
- 11 individual chart files organized in `visualizations/` subdirectories
- Charts cover core engagement, product/scent analysis, and customer insights

### Sentiment Analysis (Requires API Key)

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY='your-api-key-here'

# Run with default sample (50 comments)
python sentiment_analysis.py

# Run with custom sample size
python sentiment_analysis.py 100
```

**Outputs:**
- 4 sentiment visualization charts in `visualizations/brand_reputation/`
- `post_sentiment_analysis.csv` - Detailed post-level sentiment scores
- `brand_reputation_report.md` - Executive summary with insights

**Note:** Sentiment analysis uses Claude Sonnet 4 API (costs apply) and provides post-level reputation insights.

## Extension Proposal

- This analysis does not include data on actual conversions. Next steps of this project would
include that data, provided by Treehut, so that we could look for correlations between social media
engagement and sales to determine the most relevant signals.

- The sentiment analysis showed that promotions like PR campaigns or giveaways drive engagement, but
not necessarily positive sentiment. Next steps would be to look at the types of comments that
drive positive sentiment and focus on those types of content.

- It may be useful to build a bot that can monitor comments automatically for Treehut to notify
brand managers of comments that likely warrant a direct response.

- Provide an AI assistant to suggest language to brand managers, when authoring posts and comments, so that the language will resonate well with the target audience.

- A more thorough case should be constructed for the product recommendations made.

## AI & Tool Usage Disclosure

- Throughout the project, I have been using the Augment plugin for VS Code for interfacing with the data, as well as authoring report content and python scripting. I have been giving constant feedback, challenging assumptions, and directing attention to which parts of the project to focus on.

- The sentiment analysis script uses Claude Sonnet 4 API to analyze customer sentiment in Instagram comments for brand reputation insights.