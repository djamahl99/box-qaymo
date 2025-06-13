import sys
import os

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from pathlib import Path

from waymovqa.data.vqa_dataset import VQADataset
from waymovqa.metrics.multiple_choice import MultipleChoiceMetric
import random

from collections import defaultdict

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from collections import defaultdict
import numpy as np


# Question type mappings based on your templates
# metric = MultipleChoiceMetric()
# question_type_mappings = metric.question_type_mappings
    
question_type_mappings = {
    # Temporal Trajectory Questions
    "_prompt_faster_than_ego": ("motion", "Ego Relative Speed"),
    "_prompt_moving_towards_ego": ("motion", "Approach.."),
    "_prompt_parallel_motion": ("motion", "Parallel.."),
    "_prompt_approaching_stop_sign": ("motion", "Approaching Stop Sign"),
    "_prompt_vehicle_future_path": ("motion", "Ego Collision Prediction"),
    "_prompt_ego_following": ("motion", "Following Behavior"),
    "_movement_direction_prompt": ("motion", "Movement Direction"),
    "_speed_prompt": ("motion", "Object Speed"),

    # Instance-Referenced Questions (with bounding boxes)
    "_color_prompt": ("attribute", "Color"),
    "_label_prompt": ("attribute", "Object Type"),
    "_heading_prompt": ("attribute", "Orientation"),
    # Binary Characteristic Questions
    "_object_facing_type": ("binary", "Facing Direction"),
    "_object_movement_direction_type": ("binary", "Movement Direction"),
    # Speed category questions will be detected by pattern matching
}
    
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
    total_questions = sum(sum(counts.values()) for counts in count_per_question_type.values())
    
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
            # cat_percentage = (count / max(total_cat) * 100
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
        insidetextorientation='tangential'
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
        margin=dict(t=100, b=50, l=50, r=50)
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
        insidetextorientation='tangential'
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
        margin=dict(t=100, b=50, l=50, r=50)
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
    fig_sunburst.write_image(f'figures/question_hierarchy_sunburst_{ds.tag}.png')
    
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
    
    ds_dict = {
        "validation": (["validation", save_prefix], VQADataset(tag=f"{save_prefix}_validation")),
        "training": (["training", save_prefix], VQADataset(tag=f"{save_prefix}_training")),
        "all": ([save_prefix], VQADataset(tag=f"{save_prefix}_all")),
    }
    
    save_path = generated_samples_path / f'{save_prefix}_alldata.json'
    
    for path in generated_samples_path.rglob(f'{save_prefix}*.json'):
        if ("balance_stats" in path.name):
            continue
        
        is_relevant = False
        for keyword_list, _ in ds_dict.values():
            is_relevant |= any(kwd in path.stem for kwd in keyword_list)
                
        if is_relevant:
            ds1 = VQADataset.load_dataset(str(path))
            
            for ds_name, (keyword_list, ds) in ds_dict.items():
                is_relevant = all(kwd in path.stem for kwd in keyword_list)
                
                if is_relevant:
                    print(f'{path.stem} is relevant to {ds_name}')
                    for sample in ds1.samples:
                        ds.add_sample(sample.question, sample.answer)
                        
    for ds_name, (_, ds) in ds_dict.items():
        print('ds_name', ds_name)
        print('samples', len(ds.samples))
        
        if len(ds.samples) > 0:
            calculate_metrics(ds)
    
    
if __name__ == "__main__":
    main()