import sys
import os

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from pathlib import Path

from box_qaymo.data.vqa_dataset import VQADataset
from box_qaymo.metrics.multiple_choice import MultipleChoiceMetric
import random

from collections import defaultdict

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from collections import defaultdict
import numpy as np
import plotly.io as pio

import matplotlib.pyplot as plt
import matplotlib.patches as patches

pio.kaleido.scope.mathjax = None

# Question type mappings based on your templates
metric = MultipleChoiceMetric()
question_type_mappings = metric.question_type_mappings
    
def create_improved_log_scale(data):
    """Improved log scale with better text handling and spacing"""
    
    # Flatten all data
    all_questions = []
    all_counts = []
    all_categories = []
    
    for category, questions in data.items():
        for question_type, count in questions.items():
            all_questions.append(question_type)
            all_counts.append(count)
            all_categories.append(category)
    
    # Sort by count
    sorted_data = sorted(zip(all_questions, all_counts, all_categories), key=lambda x: x[1])
    questions, counts, categories = zip(*sorted_data)
    
    # Create shorter labels for x-axis to prevent overlap
    short_labels = []
    for q in questions:
        if len(q) > 15:
            # Create abbreviated versions for long labels
            words = q.split()
            if len(words) > 2:
                short_labels.append(f"{words[0]}<br>{' '.join(words[1:])}")
            else:
                short_labels.append(q.replace(' ', '<br>'))
        else:
            short_labels.append(q)
    
    # Color by category
    color_map = {'Binary': '#FF6B6B', 'Attribute': '#4ECDC4', 'Motion': '#45B7D1'}
    colors = [color_map[cat.title()] for cat in categories]
    
    fig = go.Figure(go.Bar(
        x=short_labels,
        y=counts,
        marker_color=colors,
        text=[f"{count:,}" for count in counts],
        textposition='outside',
        textangle=0,  # Horizontal text instead of 45 degrees
        textfont=dict(size=14, color='black'),
        hovertemplate='<b>%{customdata}</b><br>Count: %{y:,}<extra></extra>',
        customdata=questions  # Use full names in hover
    ))
    
    fig.update_layout(
        title={
            'text': "Question Type Distribution (Log Scale)",
            'x': 0.5,
            'font': {'size': 32, 'family': 'Arial', 'color': '#2c3e50'}
        },
        xaxis_title="Question Types",
        yaxis_title="Number of Questions (Log Scale)",
        yaxis_type="log",
        font=dict(size=14),
        width=1200,  # Wider to give more space
        height=700,  # Taller to accommodate text labels
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(
            b=100,  # Bottom margin for x-axis labels
            t=120,  # Top margin for text labels above bars
            l=80,   # Left margin
            r=80    # Right margin
        )
    )
    
    # Adjust y-axis range to give more space for text labels
    max_count = max(counts)
    min_count = min(counts)
    print('min_count', min_count, 'max_count', max_count)
    fig.update_yaxes(range=[np.log10(min_count + 1e-6) - 0.2, np.log10(max_count) + 0.5])
    
    # Add category legend manually with better positioning
    legend_y_pos = 0.95
    for i, (category, color) in enumerate(color_map.items()):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color=color, symbol='square'),
            name=category,
            showlegend=True
        ))
    
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        )
    )
    
    return fig

def create_horizontal_log_scale(data):
    """Alternative: Horizontal bar chart to better handle long labels"""
    
    # Flatten all data
    all_questions = []
    all_counts = []
    all_categories = []
    
    for category, questions in data.items():
        for question_type, count in questions.items():
            all_questions.append(question_type)
            all_counts.append(count)
            all_categories.append(category)
    
    # Sort by count (ascending for horizontal layout)
    sorted_data = sorted(zip(all_questions, all_counts, all_categories), key=lambda x: x[1])
    questions, counts, categories = zip(*sorted_data)
    
    # Color by category
    color_map = {'Binary': '#FF6B6B', 'Attribute': '#4ECDC4', 'Motion': '#45B7D1'}
    colors = [color_map[cat.title()] for cat in categories]
    
    fig = go.Figure(go.Bar(
        y=questions,
        x=counts,
        orientation='h',
        marker_color=colors,
        text=[f"{count:,}" for count in counts],
        textposition='outside',
        textfont=dict(size=12, color='black'),
        hovertemplate='<b>%{y}</b><br>Count: %{x:,}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "Question Type Distribution (Log Scale)",
            'x': 0.5,
            'font': {'size': 20, 'family': 'Arial', 'color': '#2c3e50'}
        },
        xaxis_title="Number of Questions (Log Scale)",
        yaxis_title="Question Types",
        xaxis_type="log",
        font=dict(size=12),
        width=1000,
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(
            l=200,  # Extra space for y-axis labels
            r=120,  # Extra space for text labels
            t=80,
            b=60
        )
    )
    
    # Add category legend
    for category, color in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color=color, symbol='square'),
            name=category,
            showlegend=True
        ))
    
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        )
    )
    
    return fig

def create_question_treemap(count_per_question_type):
    labels, parents, values, colors = [], [], [], []
    
    # Add root
    labels.append("All Questions")
    parents.append("")
    values.append(sum(sum(counts.values()) for counts in count_per_question_type.values()))
    colors.append(0)
    
    # Add data
    for hierarchy, questions in count_per_question_type.items():
        for q_type, count in questions.items():
            labels.append(f"{q_type}<br>({count:,})")
            parents.append("All Questions")
            values.append(count)
            colors.append(count)
    
    return go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        marker_colorscale="Viridis"
    ))

def  create_grouped_bars(count_per_question_type):
    fig = go.Figure()
    colors = {'motion': '#9c27b0', 'attribute': '#4caf50', 'binary': '#2196f3'}
    
    for hierarchy, questions in count_per_question_type.items():
        fig.add_trace(go.Bar(
            name=hierarchy.title(),
            x=list(questions.keys()),
            y=list(questions.values()),
            marker_color=colors[hierarchy.lower()],
            text=[f'{v:,}' for v in questions.values()],
            textposition='auto'
        ))
    
    fig.update_layout(
        font=dict(size=16, family='Arial'),
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='rgba(255,255,255,1)',
        height=900,  # Increased to accommodate perimeter labels
        width=900,
        margin=dict(t=40, b=40, l=40, r=40),  # Increased margins
        # annotations=annotations
    )
    return fig

def save_bar_chart(df: pd.DataFrame, save_path: str):
    """Generate professional bar charts suitable for academic papers (CVPR/ICCV style)"""

    if "Category" not in df.keys():
        return

    df = df.copy()

    # Replace model name (your existing code)
    actual_model_col = None
    for col in df.columns:
        if col.lower() == "model":
            actual_model_col = col
            break
    if actual_model_col:
        df[actual_model_col] = df[actual_model_col].str.replace(
            "\\textsuperscript{\\textdagger}", "†", regex=False
        )

    # Filter out unwanted categories
    df = df[df["Category"] != "overall"].copy()
    df["Category"] = df["Category"].str.title()
    
    # Professional color palette (more muted, academic-friendly)
    # Using colorbrewer colors that work well in print
    academic_colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Cyan
    ]

    models = df["Model"].unique()
    model_colors = {
        model: academic_colors[i % len(academic_colors)]
        for i, model in enumerate(models)
    }

    # Define custom category order
    category_order = ["Binary", "Attribute", "Motion"]
    
    # Filter dataframe to only include categories we want, in the order we want
    df = df[df["Category"].isin(category_order)].copy()
    
    # Convert Category to categorical with specified order
    df["Category"] = pd.Categorical(df["Category"], categories=category_order, ordered=True)

    # Create the bar chart
    fig = go.Figure()

    for model in models:
        model_df = df[df["Model"] == model].copy()

        # Group by category and get mean F1 score (in case of multiple entries per category)
        category_f1 = model_df.groupby("Category", observed=True)["F1"].mean()
        
        # Sort by the categorical order
        category_f1 = category_f1.sort_index()

        fig.add_trace(
            go.Bar(
                x=[str(cat) for cat in category_f1.index],  # Convert to string for Plotly
                y=category_f1.values,
                name=model,
                marker_color=model_colors[model],
                marker_line=dict(color="white", width=0.5),  # Subtle white borders
                opacity=0.85,
            )
        )

    # Professional academic styling
    fig.update_layout(
        # Remove title - will be added in LaTeX
        title=None,
        # Axis styling
        xaxis=dict(
            title="",
            title_font=dict(size=14, family="Arial, sans-serif"),
            tickfont=dict(size=12, family="Arial, sans-serif"),
            linecolor="black",
            linewidth=1,
            mirror=True,
            ticks="outside",
            tickwidth=1,
            ticklen=5,

        ),
        yaxis=dict(
            title="F1 Score (%)",
            title_font=dict(size=14, family="Arial, sans-serif"),
            tickfont=dict(size=12, family="Arial, sans-serif"),
            linecolor="black",
            linewidth=1,
            mirror=True,
            ticks="outside",
            tickwidth=1,
            ticklen=5,
            gridcolor="lightgray",
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1,
        ),
        # Bar grouping
        barmode="group",
        bargap=0.15,  # Gap between groups
        bargroupgap=0.1,  # Gap between bars in group
        # Legend styling
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,  # Tighter positioning
            xanchor="center",
            x=0.5,
            font=dict(size=11, family="Arial, sans-serif"),
            bgcolor="rgba(0,0,0,0)",  # Transparent background
            bordercolor="black",
            borderwidth=0,
        ),
        # Overall layout
        plot_bgcolor="white",
        paper_bgcolor="white",
        # Tighter margins for academic papers
        margin=dict(
            l=60,  # Left margin
            r=20,  # Right margin
            t=20,  # Top margin (small since no title)
            b=80,  # Bottom margin (space for legend)
        ),
        # Professional dimensions (good for 2-column papers)
        height=300,  # Shorter height
        width=500,  # Narrower width
        # Font settings
        font=dict(family="Arial, sans-serif", size=12, color="black"),
    )

    # Add subtle grid
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", gridwidth=0.5)

    # Save with high DPI for publication quality
    save_path = Path(save_path).with_suffix(".pdf")
    fig.write_image(
        save_path,
        width=500,
        height=300,
        scale=3,  # Higher scale for better quality
        format="pdf",  # PDF is preferred for LaTeX
    )
    print(f"Professional bar chart saved to: {save_path}")

def create_question_hierarchy_bars_mpl(count_per_question_type, save_path):
    """Create a professional horizontal bar chart matching academic paper style using matplotlib"""
    
    # Academic color palette matching the PDF (muted colors)
    colors = {
        'binary': '#d4e6f1',      # Light blue (like in PDF)
        'attribute': '#d5f4e6',   # Light green (like in PDF) 
        'motion': '#e8d5f4',      # Light purple (like in PDF)
    }

    # Calculate total for percentages
    total = sum(sum(counts.values()) for counts in count_per_question_type.values())
    
    # Prepare data with better organization
    y_labels = []
    percentages = []
    bar_colors = []

    plt.rcParams['font.weight'] = 'bold'
    
    # Sort hierarchy categories by total count (largest first) - or maintain specific order like PDF
    # For academic papers, you might want a specific logical order instead of size-based
    desired_order = ['binary', 'attribute', 'motion']  # Adjust as needed
    
    for hierarchy in desired_order:
        if hierarchy in count_per_question_type:
            questions = count_per_question_type[hierarchy]
            # Sort questions within each hierarchy by count (largest first)
            sorted_questions = sorted(questions.items(), key=lambda x: x[1], reverse=True)
            
            for q_type, count in sorted_questions:
                # Clean label formatting matching PDF style
                label = f"{hierarchy.title()}: {q_type.replace('_', ' ').title()}"
                y_labels.append(label)
                percentage = (count / total) * 100
                percentages.append(percentage)
                bar_colors.append(colors[hierarchy.lower()])
    
    # Reverse the order for proper top-to-bottom display (like PDF)
    y_labels = y_labels[::-1]
    percentages = percentages[::-1]
    bar_colors = bar_colors[::-1]
    
    # Create figure and axis with academic proportions
    # fig, ax = plt.subplots(figsize=(10, max(6, len(y_labels) * 0.4)))
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create horizontal bar chart
    y_positions = np.arange(len(y_labels))
    bars = ax.barh(y_positions, percentages, color=bar_colors, 
                   edgecolor='black', linewidth=0.8, alpha=1.0)
    
    # Add percentage labels on the bars (like in PDF)
    for i, (bar, percentage) in enumerate(zip(bars, percentages)):
        width = bar.get_width()
        ax.text(width + 0.2, bar.get_y() + bar.get_height()/2, 
                f'{percentage:.1f}%',
                ha='left', va='center', 
                fontsize=14, fontfamily='Arial', color='black', fontweight='bold')
    
    # Set labels and styling
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=14, fontfamily='Arial', fontweight='bold')
    ax.set_xlabel('Percentage of Questions', fontsize=15, fontfamily='Arial', fontweight='bold')
    ax.set_ylabel('Question Categories', fontsize=15, fontfamily='Arial', fontweight='bold')
    
    # Academic styling matching the PDF
    # ax.spines['top'].set_visible(True)
    # ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    # ax.spines['left'].set_visible(True)
    
    # Set spine colors and width
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1)
    
    # Grid styling (light gray, only for x-axis like in PDF)
    ax.grid(True, axis='x', color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)  # Grid behind bars
    
    # Set x-axis range and ticks
    max_percentage = max(percentages)
    ax.set_xlim(0, max_percentage * 1.15)  # Add space for labels
    
    # Set tick parameters for academic style
    ax.tick_params(axis='both', which='major', labelsize=12, 
                   width=1, length=5, direction='out', color='black')
    ax.tick_params(axis='x', which='major', bottom=True, top=False)
    
    # Create manual legend (matplotlib style)
    legend_categories = ['Binary', 'Attribute', 'Motion']
    legend_colors = [colors[cat.lower()] for cat in legend_categories]
    
    # Position legend on the right side
    legend_x = 1.02
    legend_y = 1.0
    legend_width = 0.12
    legend_height = 0.15
    
    # Create legend patches
    legend_patches = []
    for color in legend_colors:
        patch = patches.Rectangle((0, 0), 1, 1, facecolor=color, 
                                 edgecolor='black', linewidth=0.8)
        legend_patches.append(patch)
    
    # Add legend to plot
    # legend = ax.legend(legend_patches, legend_categories, 
    #                   loc='upper right', bbox_to_anchor=(1.02, 0.5),
    #                   frameon=True, fancybox=False, shadow=False,
    #                   fontsize=12, title=None)
    legend = ax.legend(legend_patches, legend_categories, 
                      loc='upper right',
                      frameon=True, fancybox=False, shadow=False,
                      fontsize=15, title=None, fontweight='bold')

    # Style legend frame
    legend.get_frame().set_linewidth(1)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_facecolor('white')
    
    # Set background color
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    # plt.subplots_adjust(right=0.85)  # Make room for legend
    
    # Save the figure at 300 DPI
    save_path = Path(save_path)
    if save_path.suffix.lower() not in ['.png', '.pdf', '.svg', '.jpg', '.jpeg']:
        save_path = save_path.with_suffix('.pdf')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Academic bar chart saved to: {save_path}")
    
    # Return figure for display if needed
    return fig

def create_question_hierarchy_bars(count_per_question_type):
    """Create a professional horizontal bar chart matching academic paper style"""
    
    fig = go.Figure()
    
    # Academic color palette matching the PDF (muted colors)
    colors = {
        'binary': '#d4e6f1',      # Light blue (like in PDF)
        'attribute': '#d5f4e6',   # Light green (like in PDF) 
        'motion': '#e8d5f4',      # Light purple (like in PDF)
    }

    # Calculate total for percentages
    total = sum(sum(counts.values()) for counts in count_per_question_type.values())
    
    # Prepare data with better organization
    y_labels = []
    values = []
    percentages = []
    bar_colors = []
    hover_texts = []
    
    # Sort hierarchy categories by total count (largest first) - or maintain specific order like PDF
    # For academic papers, you might want a specific logical order instead of size-based
    desired_order = ['binary', 'attribute', 'motion']  # Adjust as needed
    
    for hierarchy in desired_order:
        if hierarchy in count_per_question_type:
            questions = count_per_question_type[hierarchy]
            # Sort questions within each hierarchy by count (largest first)
            sorted_questions = sorted(questions.items(), key=lambda x: x[1], reverse=True)
            
            for q_type, count in sorted_questions:
                # Clean label formatting matching PDF style
                label = f"{hierarchy.title()}: {q_type.replace('_', ' ').title()}"
                y_labels.append(label)
                values.append(count)
                percentage = (count / total) * 100
                percentages.append(percentage)
                bar_colors.append(colors[hierarchy.lower()])
                
                # Simple hover text
                hover_text = f"<b>{q_type.replace('_', ' ').title()}</b><br>" + \
                            f"Count: {count:,}<br>" + \
                            f"Percentage: {percentage:.1f}%"
                hover_texts.append(hover_text)
    
    # Reverse the order for proper top-to-bottom display (like PDF)
    y_labels = y_labels[::-1]
    values = values[::-1] 
    percentages = percentages[::-1]
    bar_colors = bar_colors[::-1]
    hover_texts = hover_texts[::-1]
    
    # Create the bar chart
    fig.add_trace(go.Bar(
        x=percentages,
        y=y_labels,
        orientation='h',
        marker=dict(
            color=bar_colors,
            line=dict(color='black', width=0.8)  # Black borders like PDF
        ),
        # Percentage labels positioned like in PDF
        text=[f'{p:.1f}%' for p in percentages],
        textposition='outside',  # Position like in PDF
        textfont=dict(size=11, color='black', family='Arial, sans-serif'),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_texts,
        showlegend=False
    ))
    
    # Academic styling matching the PDF
    fig.update_layout(
        # Clean academic background
        paper_bgcolor='white',
        plot_bgcolor='white',
        
        # Academic dimensions
        height=max(400, len(y_labels) * 30 + 100),
        width=800,  # Standard academic figure width
        
        # Academic margins
        margin=dict(t=30, b=80, l=250, r=80),  # Space for labels and legend
        
        # Academic typography
        font=dict(size=12, color='black', family='Arial, sans-serif'),
        
        # X-axis styling (matching PDF)
        xaxis=dict(
            title=dict(
                text="Percentage of Questions",
                font=dict(size=14, family='Arial, sans-serif')
            ),
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.5,
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True,
            ticks='outside',
            tickwidth=1,
            ticklen=5,
            tickfont=dict(size=12, family='Arial, sans-serif'),
            range=[0, max(percentages) * 1.1],  # Adjust range based on data
            dtick=2.5  # Tick intervals like in PDF
        ),
        
        # Y-axis styling (matching PDF)
        yaxis=dict(
            title=dict(
                text="Question Categories",
                font=dict(size=14, family='Arial, sans-serif')
            ),
            showgrid=False,
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True,
            ticks='outside',
            tickwidth=1,
            ticklen=5,
            tickfont=dict(size=12, family='Arial, sans-serif'),
            categoryorder='array',
            categoryarray=y_labels
        ),
        
        # Remove default legend
        showlegend=False
    )
    
    # Add custom legend matching PDF style (positioned on the right)
    legend_categories = ['Binary', 'Attribute', 'Motion']
    legend_colors = [colors[cat.lower()] for cat in legend_categories]
    
    # Add invisible traces for clean legend
    for cat, color in zip(legend_categories, legend_colors):
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            marker=dict(
                color=color,
                line=dict(color='black', width=0.8)
            ),
            name=cat,
            showlegend=True
        ))
    
    # Style the legend to match PDF
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="right",
            x=1.02,
            font=dict(size=12, family='Arial, sans-serif'),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            itemsizing="constant"
        )
    )
    
    return fig

def create_question_hierarchy_sunburst_debug(count_per_question_type):
    """Debug version to identify why Movement Direction isn't showing"""
    
    # Prepare data for sunburst
    labels = []
    parents = []
    values = []
    colors = []
    text_labels = []
    
    # CSS-based color schemes matching your badge styles
    hierarchy_colors = {
        'motion': '#9c27b0',
        'attribute': '#4caf50', 
        'binary': '#2196f3'
    }
    
    # Calculate total for percentage calculations
    total_questions = sum(sum(counts.values()) for counts in count_per_question_type.values())
    
    print("=== DEBUGGING SUNBURST DATA ===")
    print(f"Total questions: {total_questions}")
    
    # Add root
    labels.append("All Questions")
    parents.append("")
    values.append(total_questions)
    colors.append('#ffffff')
    text_labels.append(f"<b>Total<br>{total_questions:,}</b>")
    
    # Add hierarchy categories
    for hierarchy_cat, question_types in count_per_question_type.items():
        total_cat = sum(question_types.values())
        percentage = (total_cat / total_questions) * 100
        
        print(f"\nCategory: {hierarchy_cat}")
        print(f"  Total: {total_cat} ({percentage:.1f}%)")
        
        labels.append(hierarchy_cat.title())
        parents.append("All Questions")
        values.append(total_cat)
        colors.append(hierarchy_colors[hierarchy_cat])
        text_labels.append(f"<b>{hierarchy_cat.title()}<br>{total_cat}<br>({percentage:.1f}%)</b>")
        
        # Add individual question types
        for question_type, count in question_types.items():
            total_percentage = (count / max(total_questions, 1)) * 100
            
            print(f"    {question_type}: {count} ({total_percentage:.1f}%)")
            
            labels.append(question_type)
            parents.append(hierarchy_cat.title())
            values.append(count)
            
            # Create lighter shade of parent color
            base_color = hierarchy_colors[hierarchy_cat]
            hex_color = base_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            lighter_rgb = tuple(min(255, c + 60) for c in rgb)
            lighter_color = f"#{lighter_rgb[0]:02x}{lighter_rgb[1]:02x}{lighter_rgb[2]:02x}"
            
            colors.append(lighter_color)
            question_type_nl = question_type.replace(' ', '<br>').replace('/', '<br>')
            text_labels.append(f"<b>{question_type_nl}<br>{count:,}<br>({total_percentage:.1f}%)</b>")
    
    # Print all the data arrays for debugging
    print("\n=== FINAL ARRAYS ===")
    for i, (label, parent, value, color) in enumerate(zip(labels, parents, values, colors)):
        print(f"{i}: {label} -> {parent} | {value} | {color}")
    
    # Check for duplicate labels
    print(f"\n=== DUPLICATE CHECK ===")
    print(f"Total labels: {len(labels)}")
    print(f"Unique labels: {len(set(labels))}")
    if len(labels) != len(set(labels)):
        from collections import Counter
        label_counts = Counter(labels)
        duplicates = {k: v for k, v in label_counts.items() if v > 1}
        print(f"Duplicates found: {duplicates}")
    
    # Create the figure with minimal styling to focus on data
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(
            colors=colors, 
            line=dict(color="#000000", width=1),  # Black lines for better visibility
        ),
        textfont=dict(size=8, color="black", family="Arial"),
        maxdepth=4,
        sort=False
    ))
    
    fig.update_layout(
        title="Debug: VQA Dataset Question Type Hierarchy",
        height=800,
        width=800,
        showlegend=False
    )
    
    return fig

# Also create a simple test to see what happens with just the problematic data
def test_movement_direction_only():
    """Test with minimal data to isolate the issue"""
    
    test_data = {
        'binary': {'Facing Direction': 874, 'Movement Direction': 796},
        'attribute': {'Movement Direction': 2593}  # This one should definitely show
    }
    
    labels = ["Root"]
    parents = [""]
    values = [874 + 796 + 2593]
    colors = ["#ffffff"]
    
    # Add categories
    labels.extend(["Binary", "Attribute"])
    parents.extend(["Root", "Root"])
    values.extend([874 + 796, 2593])
    colors.extend(["#2196f3", "#4caf50"])
    
    # Add individual items
    labels.extend(["Facing Direction", "Movement Direction (Binary)", "Movement Direction (Attribute)"])
    parents.extend(["Binary", "Binary", "Attribute"])
    values.extend([874, 796, 2593])
    colors.extend(["#64b5f6", "#64b5f6", "#81c784"])
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors, line=dict(color="#000000", width=1)),
        textfont=dict(size=10, color="black"),
        sort=False
    ))
    
    fig.update_layout(title="Test: Movement Direction Isolation", height=600, width=600)
    return fig

def create_question_hierarchy_sunburst(count_per_question_type):
    """Create a beautiful sunburst chart showing question type hierarchy with CSS-based colors and percentages"""
    
    # Prepare data for sunburst
    labels = []
    parents = []
    values = []
    colors = []
    text_labels = []
    cumulative_percents = []
    
    # CSS-based color schemes matching your badge styles
    hierarchy_colors = {
        'motion': '#9c27b0',
        'attribute': '#4caf50', 
        'binary': '#2196f3'
    }
    
    # Calculate total for percentage calculations
    total_questions = sum(sum(counts.values()) for counts in count_per_question_type.values())
    
    # Add root
    labels.append("All Questions")
    parents.append("")
    values.append(total_questions)
    colors.append('#ffffff')
    text_labels.append(f"<b>Total<br>{total_questions:,}</b>")
    
    # Add hierarchy categories
    for hierarchy_cat, question_types in count_per_question_type.items():
        total_cat = sum(question_types.values())
        percentage = (total_cat / max(total_questions, 1)) * 100
        
        hierarchy_cat_clean = hierarchy_cat.lower()
        
        labels.append(hierarchy_cat_clean.title())
        parents.append("All Questions")
        values.append(total_cat)
        colors.append(hierarchy_colors[hierarchy_cat_clean])
        # Simpler text for middle ring
        text_labels.append(f"<b>{hierarchy_cat_clean.title()}<br>{percentage:.1f}%</b>")
        
        # Add individual question types
        for question_type, count in question_types.items():
            total_percentage = (count / max(total_questions, 1)) * 100
            
            labels.append(question_type)
            parents.append(hierarchy_cat_clean.title())
            values.append(count)
            
            # Create lighter shade of parent color
            base_color = hierarchy_colors[hierarchy_cat_clean]
            hex_color = base_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            lighter_rgb = tuple(min(255, c + 60) for c in rgb)
            lighter_color = f"#{lighter_rgb[0]:02x}{lighter_rgb[1]:02x}{lighter_rgb[2]:02x}"
            colors.append(lighter_color)
            
            # text_labels.append(f"<b>{question_type}</b>")
            # text_labels.append(f"<b>{total_percentage:.1f}%</b>")
            text_labels.append(f"")
            
            
            # # For small percentages, show minimal text or just percentage
            # if total_percentage < 5:
            #     text_labels.append(f"<b>{total_percentage:.1f}%</b>")
            # else:
            #     question_type_short = question_type.replace(' ', '<br>')
            #     text_labels.append(f"<b>{question_type_short}</b>")
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(
            colors=colors, 
            line=dict(color="#ffffff", width=3),
            colorscale=None
        ),
        text=text_labels,
        textinfo="text",
        textfont=dict(size=14, color="black", family="Arial bold"),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage of Total: %{percentRoot:.1f}%<br>Percentage of Parent: %{percentParent:.1f}%<extra></extra>',
        maxdepth=4,
        # Key change: use 'auto' for better text positioning
        insidetextorientation='tangential'
    ))
    
    fig.update_traces(sort=False, selector=dict(type='sunburst')) 
    
    # Add annotations for outer ring labels (positioned around perimeter)
    annotations = []
    import math
    
    # Calculate cumulative angles for proper positioning
    cumulative_angle = 0
    
    print("text_labels", text_labels)
    
    for hierarchy_cat, question_types in count_per_question_type.items():
        hierarchy_cat_clean = hierarchy_cat.lower()
        total_cat = sum(question_types.values())
        
        for question_type, count in question_types.items():
            total_percentage = (count / max(total_questions, 1)) * 100
            
            # Calculate the angle for this segment
            segment_angle = (count / total_questions) * 360
            middle_angle = cumulative_angle + (segment_angle / 2)
            
            # Only add perimeter labels for small segments

            # Convert to radians and add π/2 offset to start from top
            angle_rad = math.radians(middle_angle)  # -90 to start from top
            
            # Position text outside the chart
            radius_text = 0.45  # Distance from center for text
            radius_arrow = 0.4  # Distance from center for arrow point
            
            x_pos = 0.5 + radius_text * math.cos(angle_rad)
            y_pos = 0.5 + radius_text * math.sin(angle_rad)
            
            # Arrow points to the segment
            ax_pos = 0.5 + radius_arrow * math.cos(angle_rad)
            ay_pos = 0.5 + radius_arrow * math.sin(angle_rad)
            
            
            textangle = -1*middle_angle+90
            
            # if (textangle < -90) and (textangle > -390):
            if -390 < textangle < -90:
                textangle += 180
            
            # textangle = abs(textangle)
            
            annotations.append(dict(
                x=x_pos, y=y_pos,
                # text=f"<b>{textangle:.1f}</b>",
                text=f"<b>{question_type}<br>{total_percentage:.1f}%</b>",
                # text=f"<b>{question_type}</b><br>{total_percentage:.1f}%",
                # text=f"{cumulative_angle:.2f}",
                showarrow=True,
                # textangle=textangle,
                textangle=0,
                arrowhead=0,
                arrowsize=5,
                arrowwidth=5,
                arrowcolor="#000000",
                ax=ax_pos,
                ay=ay_pos,
                xref="paper",   # Text position in paper coordinates (0-1)
                yref="paper",   # Text position in paper coordinates (0-1)
                axref="pixel",  # Arrow tail relative to text position in pixels
                ayref="pixel",  # Arrow tail relative to text position in pixels
                font=dict(size=20, color="#333333"),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#cccccc",
                borderwidth=1,
                borderpad=4,
                xanchor="center"
            ))
            
            cumulative_angle += segment_angle
    
    fig.update_layout(
        font=dict(size=24, family='Arial'),
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='rgba(255,255,255,1)',
        height=900,  # Increased to accommodate perimeter labels
        width=900,
        margin=dict(t=40, b=40, l=40, r=40),  # Increased margins
        annotations=annotations
    )
    
    return fig

def create_question_hierarchy_sunburst_adaptive_text(count_per_question_type):
    """Create sunburst with adaptive text based on segment size"""
    
    labels = []
    parents = []
    values = []
    colors = []
    text_labels = []
    
    hierarchy_colors = {
        'motion': '#9c27b0',
        'attribute': '#4caf50', 
        'binary': '#2196f3'
    }
    
    total_questions = sum(sum(counts.values()) for counts in count_per_question_type.values())
    
    # Add root
    labels.append("All Questions")
    parents.append("")
    values.append(total_questions)
    colors.append('#ffffff')
    text_labels.append(f"<b>Total<br>{total_questions:,}</b>")
    
    # Add hierarchy categories
    for hierarchy_cat, question_types in count_per_question_type.items():
        total_cat = sum(question_types.values())
        percentage = (total_cat / max(total_questions, 1)) * 100
        
        hierarchy_cat_clean = hierarchy_cat.lower()
        
        labels.append(hierarchy_cat_clean.title())
        parents.append("All Questions")
        values.append(total_cat)
        colors.append(hierarchy_colors[hierarchy_cat_clean])
        text_labels.append(f"<b>{hierarchy_cat_clean.title()}<br>{percentage:.0f}%</b>")
        
        # Add individual question types with adaptive text
        for question_type, count in question_types.items():
            total_percentage = (count / max(total_questions, 1)) * 100
            
            labels.append(question_type)
            parents.append(hierarchy_cat_clean.title())
            values.append(count)
            
            # Lighter shade
            base_color = hierarchy_colors[hierarchy_cat_clean]
            hex_color = base_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            lighter_rgb = tuple(min(255, c + 60) for c in rgb)
            lighter_color = f"#{lighter_rgb[0]:02x}{lighter_rgb[1]:02x}{lighter_rgb[2]:02x}"
            colors.append(lighter_color)
            
            # Adaptive text based on percentage size
            if total_percentage >= 10:
                # Large segments: full text
                question_type_formatted = question_type.replace('/', '/<br>')
                text_labels.append(f"<b>{question_type_formatted}<br>{total_percentage:.1f}%</b>")
            elif total_percentage >= 5:
                # Medium segments: abbreviated text
                abbreviated = question_type.replace(' Direction', '').replace(' Speed', '').replace('Movement', 'Move')
                text_labels.append(f"<b>{abbreviated}<br>{total_percentage:.1f}%</b>")
            elif total_percentage >= 2:
                # Small segments: just percentage
                text_labels.append(f"<b>{total_percentage:.1f}%</b>")
            else:
                # Very small segments: no text (rely on hover)
                text_labels.append("")
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(
            colors=colors, 
            line=dict(color="#ffffff", width=2),
            colorscale=None
        ),
        text=text_labels,
        textinfo="text",
        textfont=dict(size=13, color="black", family="Arial"),
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percentRoot:.2f}%<extra></extra>',
        maxdepth=3,
        insidetextorientation='auto'
    ))
    
    fig.update_layout(
        title={
            'text': "VQA Dataset Question Hierarchy",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 28, 'family': 'Arial Black'}
        },
        font=dict(size=13, family='Arial'),
        paper_bgcolor='white',
        height=800,
        width=800,
        margin=dict(t=70, b=50, l=50, r=50),
        showlegend=False
    )
    
    return fig

def create_question_answer_hierarchy_sunburst(n_samples_per_question_name_answer, count_per_question_type):
    """Create a beautiful sunburst chart showing question type hierarchy with CSS-based colors and percentages"""
    
    # Prepare data for sunburst
    labels = []
    parents = []
    values = []
    colors = []
    text_labels = []
    
    # CSS-based color schemes matching your badge styles
    css_color_schemes = {
        'info': '#2196f3',      # Blue - for information type questions
        'attr': '#e91e63',      # Pink - for attribute questions  
        'hog': '#9c27b0',       # Purple - for higher-order questions
        'inst': '#4caf50',      # Green - for instruction questions
        'cat': '#ff9800',       # Orange - for category questions
        'mov': '#f44336',       # Red - for movement questions
        'app': '#009688',       # Teal - for application questions
        'rel': '#8bc34a'        # Light green - for relational questions
    }
    
    # Map your hierarchy categories to CSS colors
    hierarchy_colors = {
        'motion': '#9c27b0',
        'attribute': '#4caf50', 
        'binary': '#2196f3'
    }
    
    # Calculate total for percentage calculations
    total_questions = sum(sum(counts.values()) for counts in n_samples_per_question_name_answer.values())
    
    # Add root
    labels.append("All Questions")
    parents.append("")
    values.append(total_questions)
    # colors.append('#37474f')  # Dark blue-gray for root
    colors.append('#ffffff')  # Dark blue-gray for root
    text_labels.append(f"<b>Total<br>{total_questions:,}</b>")
    
    # Add hierarchy categories
    for hierarchy_cat, question_types in count_per_question_type.items():
        total_cat = sum(question_types.values())
        percentage = (total_cat / max(total_questions, 1)) * 100
        
        labels.append(hierarchy_cat.title())
        parents.append("All Questions")
        values.append(total_cat)
        colors.append(hierarchy_colors[hierarchy_cat])
        text_labels.append(f"<b>{hierarchy_cat.title()}<br>{total_cat}<br>({percentage:.1f}%)</b>")
        
        # Add individual question types with lighter shades
        for question_type, count in question_types.items():
            cat_percentage = (count / total_cat) * 100
            total_percentage = (count / max(total_questions, 1)) * 100
            
            labels.append(question_type)
            parents.append(hierarchy_cat.title())
            values.append(count)
            
            # Create lighter shade of parent color
            base_color = hierarchy_colors[hierarchy_cat]
            # Convert hex to RGB and lighten
            hex_color = base_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            # Lighten by adding to each RGB component
            lighter_rgb = tuple(min(255, c + 60) for c in rgb)
            lighter_color = f"#{lighter_rgb[0]:02x}{lighter_rgb[1]:02x}{lighter_rgb[2]:02x}"
            
            colors.append(lighter_color)
            question_type_nl = question_type.replace(' ', '<br>').replace('/', '/<br>')
            
            text_labels.append(f"<b>{question_type_nl}<br>{count:,}<br>({total_percentage:.1f}%)</b>")

            if hierarchy_cat.lower() == "binary":
                print('question_type, count', question_type, count)
                print(f"<b>{question_type_nl}<br>{count:,}<br>({total_percentage:.1f}%)</b>")
        # print('text_labels', text_labels)
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(
            colors=colors, 
            line=dict(color="#ffffff", width=3),
            colorscale=None
        ),
        text=text_labels,
        textinfo="text",
        textfont=dict(size=16, color="black", family="Arial"),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage of Total: %{percentRoot:.1f}%<br>Percentage of Parent: %{percentParent:.1f}%<extra></extra>',
        maxdepth=4,
        # insidetextorientation='radial'
        # insidetextorientation='tangential'
        # insidetextorientation='auto'
    ))
    
    fig.update_layout(
        title={
            'text': "VQA Dataset Question Hierarchy",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 40, 'family': 'Arial Black', 'color': '#000000'}
        },
        font=dict(size=16, family='Arial'),
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='rgba(255,255,255,1)',
        height=800,
        width=800,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def create_answer_distribution_chart(n_samples_per_answer):
    """Create a beautiful donut chart for top answers"""
    
    # Get top 15 answers
    sorted_answers = sorted(n_samples_per_answer.items(), key=lambda x: x[1], reverse=True)[:15]
    answers, counts = zip(*sorted_answers)
    
    # Beautiful color palette
    colors = px.colors.qualitative.Set3[:len(answers)]
    
    fig = go.Figure(go.Pie(
        labels=answers,
        values=counts,
        hole=0.4,
        marker=dict(
            colors=colors,
            line=dict(color='white', width=2)
        ),
        textinfo='label+percent',
        textposition='auto',
        textfont=dict(size=12, family='Arial'),
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
    ))
    
    fig.add_annotation(
        text=f"Total<br>{sum(counts):,}<br>Answers",
        x=0.5, y=0.5,
        font_size=16,
        font_family="Arial Bold",
        showarrow=False
    )
    
    fig.update_layout(
        title={
            'text': "Top 15 Answer Distribution",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'family': 'Arial Black'}
        },
        font=dict(size=14),
        paper_bgcolor='rgba(0,0,0,0)',
        height=600,
        margin=dict(t=80, b=20, l=20, r=20),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    
    
    return fig

def calculate_metrics(ds: VQADataset):
    n_samples_per_question_name = defaultdict(int)
    n_samples_per_answer = defaultdict(int)
    n_samples_per_scene_id = defaultdict(int)
    n_samples_per_generator_name = defaultdict(int)
    n_samples_per_camera_name = defaultdict(int)
    n_samples_per_question = defaultdict(int)
    n_samples_per_question_name_answer = defaultdict(int)
        
    for sample in ds.samples:
        n_samples_per_question_name[sample.question.question_name] += 1
        n_samples_per_answer[sample.answer.answer] += 1
        n_samples_per_scene_id[sample.question.scene_id] += 1
        if hasattr(sample.question, 'camera_name'):
            n_samples_per_camera_name[sample.question.camera_name] += 1
        n_samples_per_generator_name[sample.question.generator_name] += 1
        n_samples_per_question[sample.question.question] += 1

        question_name = sample.question.question_name

        if question_name in question_type_mappings:
            hierarchy_cat, question_type = question_type_mappings[question_name]
            n_samples_per_question_name_answer[(hierarchy_cat, question_type, sample.answer.answer)] += 1

    count_per_question_type = {}
    for (hierarchy_cat, question_type) in question_type_mappings.values():
        count_per_question_type.setdefault(hierarchy_cat, {})
        count_per_question_type[hierarchy_cat].setdefault(question_type, 0)
        
    
    for question_name, count in n_samples_per_question_name.items():
        if question_name in question_type_mappings:
            hierarchy_cat, question_type = question_type_mappings[question_name]
            count_per_question_type[hierarchy_cat][question_type] += count
    
    res = dict(
        n_samples_per_question_name=n_samples_per_question_name,
        n_samples_per_answer=n_samples_per_answer,
        n_samples_per_scene_id=n_samples_per_scene_id,
        n_samples_per_camera_name=n_samples_per_camera_name,
        n_samples_per_generator_name=n_samples_per_generator_name,
        # n_samples_per_question=n_samples_per_question,
        count_per_question_type=count_per_question_type
    )
    

    
    # 1. Question Type Distribution (Hierarchical Sunburst)
    fig_sunburst = create_question_hierarchy_sunburst(count_per_question_type)
    # fig_sunburst.write_image(f'figures/question_hierarchy_sunburst_{ds.tag}.png')
    # Save with high DPI for publication quality
    fig_sunburst.write_image(
        f'figures/question_hierarchy_sunburst_{ds.tag}.pdf',
        scale=3,  # Higher scale for better quality
        format="pdf",  # PDF is preferred for LaTeX
    )
    
    # 1. Question Type Distribution (Hierarchical Sunburst)
    fig_sunburst = create_question_hierarchy_sunburst_adaptive_text(count_per_question_type)
    fig_sunburst.write_image(
        f'figures/question_hierarchy_sunburst_adaptive_{ds.tag}.pdf',
        scale=3,  # Higher scale for better quality
        format="pdf",  # PDF is preferred for LaTeX
    )
    
    fig = create_question_hierarchy_bars(count_per_question_type)
    fig.write_image(
        f'figures/create_question_hierarchy_bars_{ds.tag}.pdf',
        scale=3,  # Higher scale for better quality
        format="pdf",  # PDF is preferred for LaTeX
    )
    
    create_question_hierarchy_bars_mpl(count_per_question_type, f'figures/create_question_hierarchy_bars_{ds.tag}_mpl.pdf')

    fig = create_grouped_bars(count_per_question_type)
    fig.write_image(
        f'figures/create_grouped_bars_{ds.tag}.pdf',
        scale=3,  # Higher scale for better quality
        format="pdf",  # PDF is preferred for LaTeX
    )
    
    fig = create_question_treemap(count_per_question_type)
    fig.write_image(
        f'figures/create_question_treemap_{ds.tag}.pdf',
        scale=3,  # Higher scale for better quality
        format="pdf",  # PDF is preferred for LaTeX
    )
    
    # 1. Question Type Distribution (Hierarchical Sunburst)
    fig_sunburst = create_question_hierarchy_sunburst_debug(count_per_question_type)
    fig_sunburst.write_image(f'question_hierarchy_sunburst_debug_{ds.tag}.png')
    
    # 3. Answer Distribution (Top answers only)
    # fig_answers = create_answer_distribution_chart(n_samples_per_answer)
    # fig_answers.write_image('answer_distribution_chart.png')
    
    # # 3. Answer Distribution (Top answers only)
    # fig_answers = create_improved_log_scale(count_per_question_type)
    # fig_answers.write_image('create_improved_log_scale.png')
    
    # fig_answers = create_horizontal_log_scale(count_per_question_type)
    # fig_answers.write_image('create_horizontal_log_scale.png')
    
    with open(f"figures/dataset_statistics_{ds.tag}.json", 'w') as f:
        json.dump(res, f)

def main():
    generated_samples_path = Path("/media/local-data/uqdetche/waymo_vqa_dataset/generated_vqa_samples/")
    
    random.seed(42)
    
    save_prefix = "26_05_2025_export"
    
    ds =  VQADataset(tag=f"{save_prefix}_all")
    
    # ds_dict = {
    #     "validation": (["validation", save_prefix], VQADataset(tag=f"{save_prefix}_validation")),
    #     "training": (["training", save_prefix], VQADataset(tag=f"{save_prefix}_training")),
    #     "all": ([save_prefix], VQADataset(tag=f"{save_prefix}_all")),
    # }
    
    # save_path = generated_samples_path / f'{save_prefix}_alldata.json'
    
    question_ids_seen = set()
    
    prompt_generators = ["ObjectDrawnBoxPromptGenerator", "ObjectBinaryPromptGenerator", "EgoRelativeObjectTrajectoryPromptGenerator"]
    
    for prompt_generator in prompt_generators:
        for split in ["training", "validation"]:
            path = generated_samples_path / f"{save_prefix}_{prompt_generator}_{split}.json"

            ds1 = VQADataset.load_dataset(str(path))
            
            print("path", path.name)
            
            for sample in ds1.samples:
                if sample.question.question_id not in question_ids_seen:
                    ds.add_sample(sample.question, sample.answer)
                    question_ids_seen.add(sample.question.question_id)
        
        # is_relevant = False
        # for keyword_list, _ in ds_dict.values():
        #     is_relevant |= any(kwd in path.stem for kwd in keyword_list)
                
        # if is_relevant:
        #     ds1 = VQADataset.load_dataset(str(path))
            
        #     for ds_name, (keyword_list, ds) in ds_dict.items():
        #         is_relevant = all(kwd in path.stem for kwd in keyword_list)
                
        #         if is_relevant:
        #             print(f'{path.stem} is relevant to {ds_name}')
        #             for sample in ds1.samples:
        #                 ds.add_sample(sample.question, sample.answer)
          
    calculate_metrics(ds)
    
    
if __name__ == "__main__":
    main()