# Community Environments

This directory is home to community-contributed training environments for Atropos. Environments submitted by the community will be placed here after an initial code review.

**Note:** Environments in this directory are pending full testing and integration. While they have passed a basic code check, they may not yet have been rigorously validated on our compute cluster.

## Contributing Your Environment

We encourage you to contribute your own RL environments! When developing a new environment, please follow these guidelines:

1.  **Create your environment in this `environments/community/` subdirectory.** This helps us keep new submissions organized.
2.  **Preferred Import Style:** We prefer that you treat your environment's directory as the package root for imports within your environment code. For example, if you need to import `SomeClass`, you can do so directly:
    ```python
    from some_file_in_my_env import SomeClass
    ```
    This helps maintain consistency and makes it easier to integrate your environment.

Once your environment is ready, please follow the guidelines in our main [CONTRIBUTING.md](../../../CONTRIBUTING.md) to submit your contribution. We look forward to seeing what you build!
