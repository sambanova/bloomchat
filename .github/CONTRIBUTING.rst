##################
Contributing Guide
##################

#. Any code changes should come with accompanying JIRA ticket.
#. Open a Pull Request (PR) to submit any code changes
#. Please make sure your local environment is set-up properly to run preliminary checks

Local Environment Setup
***********************

**NOTE**: Please ensure that your Python version matches the version used in CI flow in ``.circleci/config.yml`` file.

#. Create Python virtual environment using ``pipenv``

    .. code-block::

        pip install pipenv
        pipenv --python <VERSION>  # Creates a virtual environment for the project with specified VERSION; e.g. pipenv --python 3.9

#. Install and set-up Required Python Packages in editable mode

    .. code-block::

        pipenv run pip install -e .
        pipenv sync --categories=default,build-packages,dev-packages,docs-packages,tests-packages
        pipenv --help

#. Initialize Pre-commit

    .. code-block::

        pipenv run pre-commit install

#. To run any Python commands, you should either be in ``pipenv`` shell (``pipenv shell`` to enter) or use ``pipenv run`` in front of the command

    .. code-block::

        # Example to run pytest
        pipenv run pytest

        # OR
        pipenv shell
        pytest

#. If you update ``setup.cfg``, or ``pyproject.toml``

   - ``requirements`` files would need to be regenrated

     - For this, you would need to have ``docker`` installed on your machine.
   - ``Pipfile.lock`` would need to be regenerated

     .. code-block::

        pipenv run pre-commit run --all-files --hook-stage manual pipenv-lock

Important Python Versions
*************************

Python versions are defined in these places:

- ``pyproject.toml``
   Defines the python-version requirement of the project
- ``Pipfile``
   Defines python-version used to configure ``pipenv``
- ``.circleci/config.yml``
   Python-version used in CI flow

**NOTE**: When updating ``python`` version for; ensure that all ``pyproject.toml``, ``Pipfile``, and ``.circleci/config.yml`` are in sync.

Naming Conventions
******************

#. git branch naming convention

   - ``<username>/<feature/bugfix/hotfix>/<ABC-1234-a-short-and-clear-description>``

   - e.g. ``ranjanl/feature/BTD-1121-json-tests-should-support-iommu``


Pull Request (PR) Process
*************************

#. Ensure ``pre-commit`` is running with the repository configuration before opening a PR
#. A PR should only contain one unit of work; please open multiple PR's as necessary
#. Do your best to make sure all PR checkboxes could be ticked off
#. The PR should pass all the automated checks before it could be merged

Pull Request (PR) Review
************************

#. If you are assigned to review a PR, respond as soon as possible
   - If you are not the right person to be reviewing the PR, please find another relevant person from your team and assign it to them
#. Provide actionable explicit comments with code-examples if possible
#. For soft suggestions use prefix ``nit:`` in your comments
#. Use ``Start Review`` feature to submit multiple comments at once.
#. Use ``Request Changes`` to block the PR explicitly until the questions/concerns are resolved.

Code of Conduct
***************

#. When reviewing PR, imagine yourself as a PR submitter
#. When responding to PR feedback, imagine yourself as a PR reviewer
#. Be honest, direct, and respectful in your communication; embrace difference of opinions
#. For any comments that is going through many back and forths; hop on a quick-call to understand the other persons viewpoint
