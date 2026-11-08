# Prompt log

Every prompt given to AI tools for this task, in order, with a one-line note on what I did with the answer. Wrong or rejected answers are marked.

## 1 — claude.ai (planning chat), Nov 8
> The project I am supposed to do is in this repo: https://github.com/Abhivanth-yellow/quickcart-intent-task. Go through the README file, understand the problem statement, the data and the expectations for completing the project. Now, with all that, prepare a proper implementation plan (as a document) to complete the project. Besides completing the project, I want proper notes that should be updated throughout the project about everything I need to know. Completing the project alone is not enough; I should be completely able to understand every decision made throughout the project. At the end (in your reply), let me know how to actually implement the project, like using ClaudeCode, AG, etc., as I need it to be done by tonight at 7 pm and need a complete understanding of the project and every decision we make throughout the project. Do the things as required, and let me know how I can start the work now. Also, will add this chat to a Claude project, so all the chats related to this project can stay together and share knowledge.

### Notes: 
used the above prompt providing the repo link to claude to prepare an implementation plan to learn alongside building. It gave me 4 different files to provide to ClaudeCode and start working. Also a project overview file to upload to Claude Project so that everry chat will have the context regarding the project. Claude divided the work into 9 phases to complete the project. Also, it mentioned all the `11` errors/mistakes it had found to cover in the implementation phases. 


## 2 - claude.ai (planning chat), Nov 8
> Do you want me to commit these 4 files to the repo?? I don't think all of them are required to be in the repo. Only the Claude.md file is enough, right? Also, let me know if you want me to add the implementation plan doc to my local repo so Claude Code can use it. I forked the repo, cloned it, added the 4 files above to my local clone, created a venv, and configured Claude Code to start working. Can I start work, or is there anything I should check first? Let me know everything needed and will start the work now.

### Notes: 
With Claude's suggestion, pushed all the planning files to repository and setup the environemnt for ClaudeCode to work on the project. Also, an immplementation plan file is explicitly made for ClaudeCode to work on the project. 


## 3 - Claude Code (VS Code), Nov 8
> Read README.md, CLAUDE.md and docs/PLAN.md. The fork already contains the starter commits, so skip the "starter as received" commit. I even committed the planning files, so no need to commit them now. Do Phase 0: confirm the baseline numbers. Then propose Phase 1 and wait for go.

### Notes: 
In the planning phase, claude mentioned the errors to be corrected in Phaase 1, will correct those fixes in the next step. 
-> Calculating the actual `Macro F1 score` where it is mentioned as `Macro F1 score` and actually calculated a weighted score. 
-> Confusion-Matrix doesn't have labels.


## 4 - Claude Code (VS Code), Nov 8
> go

### Notes:
claude wrote the whole code as mentioned as phase-1 in the `docs/plan.md` file. Now, I need to check the notes and decisions it made in this step.


## 5 - Claude Code (VS Code), Nov 8
> Before proceeding to phase 2, what all do you want me to read and understand in prior. Also, which open-source model are we picking for this task and why? I need a proper reasoning. Let me know if there is anything much important to consider from phase 1 run which might change the implementation steps later.

### Notes:
Found the realted notes to the implementations in phase 1 and gone through all of them and returned with some questions to claude.

## 6 - Claude Code (VS Code), Nov 8
> I have some questions before proceeding:
> 1) the statement about the models mentioned in the Readme file was: "Hosted APIs that serve open-source models are allowed." I think we can use heavier model through API services which are open-sourced. I think we just don't need to stick to this 7B model. check it out and come up with a decision and proper reasoning. Let me know if I am wrong.
> 2) In 1st table in results/eval_bugs.md, in point number 5, you mentioned that the actual `out_of_scope` rows as 378. Did you actually counted it from data/dev.csv??
> 3) In table 1 in results/eval_bugs.md, explain me the point numbers 7 and 9 properly. I didnt get them.
> 4) what does the results/baseline_fixed.csv actually represents?
> 5) You said you want me to label manually. What exactly do you want me to do? Be precise and explain clearly.
> 6) Do oversampling only if it is required. Give me proper reasoning if it is actually required.
> 7) How do you even found that the dev data leaked into vectoriser? And how did you observe the numbber 0.4303?
>
> on your questions:
> 1) let me know what you actually mean. I mentioned my query above
> 2) mentioned above
> 3) mentioned above
> 4) I dont have GPU access.

### Notes:
got answers for my queries, need to start phase 2 with 2 design decisions:
-> Not labelling the records manually
-> No oversampling of the data for Hinglish and Multi-request examples


## 7 - Claude Code (VS Code), Nov 8
> I dont prefer manually labelling 250 exmaples. It is not part of the task and it is a waste of time. you asked me about the promo_code_issue in reply for quesiton 6. Think if it is actually contributes to the requirement of the project? If so, do it and give me the proper reasoning.
> 1) api key is present in .env file with name 'GROQ_API_KEY'.
> 2) no blind labelling
> 3) mentioned above
> 4) we will run the pilot
>
> you can start working on phase 2.

### Notes:
claude written code for phase 2 partially to flag suspicious labels with Five-fold cross-validated disagreement on train. The Groq server returned that the API key as invalid one. SO I created a new key and pasted it back in .env. Claude cannot access .env file.


## 8 - Claude Code (VS Code), Nov 8
> I created a new key and pasted in .env file. you can check it now. Also, before commiting the code, I want you to guide me exactly what to read and understand regarding phase 2 implementation, so that we can proceed further.

### Notes:
the api key was still returning the invalid request. It might be because of stale memory which has not been overwritten with the new key. Will try with `load_dotenv(dotenv_path=env_path, override=True)` in code to overwrite the value in memory with the new value. 

## 9 - Claude Code (VS Code), Nov 8
> I changed the new key in the .env even before. It might not be working because of the stale memory on disk. This happened to me before. Try it in code with `load_dotenv(dotenv_path=env_path, override=True)` to overwrite the existing stale memory.
>
> on your quesitons:
> 1) Yes, I agree with that and changed the CLAUDE.md file accordingly.
> 2) Yes, I agree
>
> we can proceed next.

### Notes:
The key worked now, and the two models Qwen3.8 - 27B and GPT-OSS - 120B are used to predict the labels for `dev_sample.csv`. Claude created a new file `dev_review.csv` where the models disagree with actual silver labels. I manually labelled them to check the Wilson interval.

## 10 - Claude Code (VS Code), Nov 8
> I changed the column `adi_label` to `my_label`. So, make the changes wherever required accordingly. And done with what you had asked me to do. Manually labelled all the 50 with the error_type where my_label differs with silver_label. Proceed next.

### Notes:
a sample in `dev_sample.csv` was labelled wrong and corrected it



## 11 - Claude Code (VS Code), Nov 8
> made the requried changes. proceed for the next steps.

Note: TODO (Adi)

## 12 - Claude Code (VS Code), Nov 8
> go

Note: TODO (Adi)

## 13 - Claude Code (VS Code), Nov 8
> go

Note: TODO (Adi)

## 14 - Claude Code (VS Code), Nov 8
> go

Note: TODO (Adi)
