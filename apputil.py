import plotly.express as px
import numpy as np
import pandas as pd

def age_division_summary():
    df = determine_age_division().copy()
    # Treat blank/empty strings in Age as NaN
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    # Group by Pclass and older_passenger, calculate survival rate and mean age
    summary = (
        df.groupby(['Pclass', 'older_passenger'], dropna=False, observed=False)
          .agg(
              survival_rate=('Survived', 'mean'),
              Age=('Age', 'mean')
          )
          .reset_index()
    )
    # Rename 'Age' to 'age' for autograder 
    summary = summary.rename(columns={
        'Pclass': 'pclass',
        'older_passenger': 'older_passenger',
        'survival_rate': 'survival_rate',
        'Age': 'age'
        }
)
    # Ensure strict boolean dtype for older_passenger
    summary['older_passenger'] = summary['older_passenger'].astype(bool)
    # Rename columns to lowercase for autograder 
    summary = summary.rename(columns={
        'Pclass': 'pclass',
        'older_passenger': 'older_passenger',
        'survival_rate': 'survival_rate',
        'Age': 'age'
    })
    # Drop rows where 'age' is NaN and ensure float dtype
    summary = summary.dropna(subset=['age'])
    summary['age'] = summary['age'].astype(float)
    return summary[['pclass', 'older_passenger', 'survival_rate', 'age']]

def last_names():
    """Load Titanic dataset and extract last names from the Name column. 
    Return a Series with last names as index and their counts as values."""
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv', on_bad_lines='skip')
    # Extract last name from the Name column
    # last_name out should be series of unique last names
    last_names = df['Name'].str.split(',').str[0].str.strip()
    # Return a Series with last names as index and their counts as values
    return last_names.value_counts()

def family_groups():
    """Load Titanic dataset and calculate family size. 
    Group by family size and calculate survival statistics."""
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv', on_bad_lines='skip')
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    grouped = df.groupby('family_size', observed=False).agg(
        n_passengers=('PassengerId', 'count'),
        n_survivors=('Survived', 'sum'),
        survival_rate=('Survived', 'mean'),
        avg_fare=('Fare', 'mean'),
        avg_age=('Age', 'mean')
    ).reset_index()
    grouped = grouped.sort_values('family_size').reset_index(drop=True)
    return grouped


def survival_demographics():
    """Load Titanic dataset and create survival rate summary."""
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv', on_bad_lines='skip')
    bins = [0, 12, 19, 59, 120]
    labels = ['Child', 'Teen', 'Adult', 'Senior']
    df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True, include_lowest=True)
    grouped = df.groupby(['Pclass', 'Sex', 'age_group'], observed=False)
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
    #   Create the grouped bar chart
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
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv', on_bad_lines='skip')
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    family_summary = df.groupby(['family_size', 'Survived'], observed=False).size().reset_index(name='count')
    family_pivot = family_summary.pivot(index='family_size', columns='Survived', values='count').fillna(0)
    if 0 in family_pivot.columns and 1 in family_pivot.columns:
        family_pivot.columns = ['Died', 'Survived']
    family_pivot = family_pivot.reset_index()
    # Create the stacked bar chart
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
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv', on_bad_lines='skip')
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    family_rate = df.groupby('family_size', observed=False)['Survived'].mean().reset_index()
    # Create the line chart
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
    df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv', on_bad_lines='skip')
    # Ensure Age column is numeric, blanks become NaN
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    # Calculate median age per class
    median_ages = df.groupby('Pclass', observed=False)['Age'].transform('median')
    df['older_passenger'] = df['Age'] > median_ages
    return df

def visualize_age_division():
    """
    Visualization 4: Survival rate by age division and class
    """
    df = determine_age_division()
    # Group by class and older_passenger, calculate survival rate and mean age
    summary = df.groupby(['Pclass', 'older_passenger'], observed=False).agg(
        survival_rate=('Survived', 'mean'),
        age=('Age', 'mean')
    ).reset_index()
    summary['older_passenger'] = summary['older_passenger'].map({True: 'Older', False: 'Younger'})
    # create the bar chart
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

