import plotly.express as px
import numpy as np
import pandas as pd

# Titanic survival demographics
def survival_demographics():
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
    return summary

# Visualization 1: Survival demographics
def visualize_demographic():
    data = survival_demographics()
    fig = px.bar(
        data,
        x='age_group',
        y='survival_rate',
        color='Sex',
        barmode='group',
        facet_col='Pclass',
        category_orders={
            'age_group': ['Child', 'Teen', 'Adult', 'Senior'],
            'Sex': ['female', 'male'],
            'Pclass': [1, 2, 3]
        },
        labels={
            'age_group': 'Age Group',
            'survival_rate': 'Survival Rate',
            'Sex': 'Sex',
            'Pclass': 'Passenger Class'
        },
        title='Survival Rate by Class, Sex, and Age Group'
    )
    fig.update_layout(yaxis_tickformat='.0%')
    return fig

# Visualization 2: Survival by family size (stacked bar)
def visualize_families():
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

# Visualization 3: Survival rate by family size (line)

def visualize_family_size():
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

# Bonus: Age division by class
def determine_age_division():
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
    # Calculate median age per class
    median_ages = df.groupby('Pclass')['Age'].transform('median')
    df['older_passenger'] = df['Age'] > median_ages
    return df

def visualize_age_division():
    df = determine_age_division()
    # Group by class and older_passenger, calculate survival rate
    summary = df.groupby(['Pclass', 'older_passenger'])['Survived'].mean().reset_index()
    summary['older_passenger'] = summary['older_passenger'].map({True: 'Older', False: 'Younger'})
    fig = px.bar(
        summary,
        x='older_passenger',
        y='Survived',
        color='older_passenger',
        facet_col='Pclass',
        labels={'Survived': 'Survival Rate', 'older_passenger': 'Age Division', 'Pclass': 'Passenger Class'},
        title='Survival Rate by Age Division and Class'
    )
    fig.update_layout(yaxis_tickformat='.0%')
    return fig

