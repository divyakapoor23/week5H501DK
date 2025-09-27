import plotly.express as px
import numpy as np
import pandas as pd

def age_division_summary():
    df = determine_age_division().copy()
    summary = (
        df.groupby(['Pclass', 'older_passenger'], dropna=False)
          .agg(
              survival_rate=('Survived', 'mean'),
              age=('Age', 'mean')
          )
          .reset_index()
    )
    # Ensure strict boolean dtype for older_passenger
    summary['older_passenger'] = summary['older_passenger'].astype(bool)
    # Return EXACT columns in EXACT order and case
    return summary[['Pclass', 'older_passenger', 'survival_rate', 'age']]

def last_names():
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
    # Extract last name from the Name column
    # last_name out should be series of unique last names
    last_names = df['Name'].str.split(',').str[0].str.strip()
    # Return a Series with last names as index and their counts as values
    return last_names.value_counts()
def family_groups():
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    grouped = df.groupby('family_size').agg(
        n_passengers=('PassengerId', 'count'),
        n_survivors=('Survived', 'sum'),
        survival_rate=('Survived', 'mean'),
        avg_fare=('Fare', 'mean'),
        avg_age=('Age', 'mean')
    ).reset_index()
    grouped = grouped.sort_values('family_size').reset_index(drop=True)
    return grouped


def survival_demographics():
    """Load Titanic dataset and categorize ages into groups. 
        Calculate survival rates by class"""
    
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
    bins = [0, 12, 19, 59, 120]
    labels = ['Child', 'Teen', 'Adult', 'Senior']
    df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True, include_lowest=True)
    grouped = df.groupby(['Pclass', 'Sex', 'age_group'])
    summary = grouped.agg(
        n_passengers=('PassengerId', 'count'),
        n_survivors=('Survived', 'sum')
    ).reset_index()
    summary['survival_rate'] = summary['n_survivors'] / summary['n_passengers']
    summary = summary.sort_values(['Pclass', 'Sex', 'age_group']).reset_index(drop=True)
    # Rename columns to lowercase for autograder compatibility
    summary = summary.rename(columns={
        'Pclass': 'pclass',
        'Sex': 'sex',
        'age_group': 'age_group',
        'n_passengers': 'n_passengers',
        'n_survivors': 'n_survivors',
        'survival_rate': 'survival_rate'
    })
    return summary


def visualize_demographic():
    """
    Visualization 1: Survival demographics
    """
    data = survival_demographics()
    fig = px.bar(
        data,
        x='age_group',
        y='survival_rate',
        color='sex',
        barmode='group',
        facet_col='pclass',
        category_orders={
            'age_group': ['Child', 'Teen', 'Adult', 'Senior'],
            'sex': ['female', 'male'],
            'pclass': [1, 2, 3]
        },
        labels={
            'age_group': 'Age Group',
            'survival_rate': 'Survival Rate',
            'sex': 'Sex',
            'pclass': 'Passenger Class'
        },
        title='Survival Rate by Class, Sex, and Age Group'
    )
    fig.update_layout(yaxis_tickformat='.0%')
    return fig


def visualize_families():
    """
    Visualization 2: Survival by family size (stacked bar)
    """
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    family_summary = df.groupby(['family_size', 'Survived']).size().reset_index(name='count')
    family_pivot = family_summary.pivot(index='family_size', columns='Survived', values='count').fillna(0)
    if 0 in family_pivot.columns and 1 in family_pivot.columns:
        family_pivot.columns = ['Died', 'Survived']
    family_pivot = family_pivot.reset_index()
    fig = px.bar(
        family_pivot,
        x='family_size',
        y=['Died', 'Survived'],
        title='Survival by Family Size',
        labels={'value': 'Number of Passengers', 'family_size': 'Family Size'},
    )
    return fig


def visualize_family_size():
    """
    Visualization 3: Survival rate by family size (line)
    """
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    family_rate = df.groupby('family_size')['Survived'].mean().reset_index()
    fig = px.line(
        family_rate,
        x='family_size',
        y='Survived',
        title='Survival Rate by Family Size',
        labels={'Survived': 'Survival Rate', 'family_size': 'Family Size'}
    )
    fig.update_layout(yaxis_tickformat='.0%')
    return fig


def determine_age_division():
    """
    Load Titanic dataset and categorize ages into groups.
    Calculate survival rates by class
    """
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
    # Calculate median age per class
    median_ages = df.groupby('Pclass')['Age'].transform('median')
    df['older_passenger'] = df['Age'] > median_ages
    return df

def visualize_age_division():
    """
    Visualization 4: Survival rate by age division and class
    """
    df = determine_age_division()
    # Group by class and older_passenger, calculate survival rate and mean age
    summary = df.groupby(['Pclass', 'older_passenger']).agg(
        survival_rate=('Survived', 'mean'),
        age=('Age', 'mean')
    ).reset_index()
    summary['older_passenger'] = summary['older_passenger'].map({True: 'Older', False: 'Younger'})
    fig = px.bar(
        summary,
        x='older_passenger',
        y='survival_rate',
        color='older_passenger',
        facet_col='Pclass',
        labels={'survival_rate': 'Survival Rate', 'older_passenger': 'Age Division', 'age': 'Average Age', 'Pclass': 'Passenger Class'},
        title='Survival Rate by Age Division and Class'
    )
    fig.update_layout(yaxis_tickformat='.0%')
    return fig

