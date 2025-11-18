# FLOSS/Fund Application Guide for Semantica

## Overview

FLOSS/fund is a $1 million per year fund by Zerodha supporting free and open source projects globally. They provide up to $100,000 per year per project.

## Application Process

### Step 1: Create funding.json Manifest

A `funding.json` file has been created in the repository root. This file:
- Follows the fundingjson.org standard
- Contains project information, funding request, impact, goals, and budget
- Requests $50,000 USD per year (fits within FLOSS/fund's $25,000 increment structure)

### Step 2: Make funding.json Publicly Accessible

The `funding.json` file needs to be accessible via a public URL. Options:

**Option A: GitHub Raw URL (Recommended)**
```
https://raw.githubusercontent.com/Hawksight-AI/semantica/main/funding.json
```

**Option B: Project Website**
If you have a project website, host it there at:
```
https://your-domain.com/.well-known/funding.json
```

### Step 3: Verify Project Association (Optional but Recommended)

To verify the association between your project and the funding manifest, implement the `wellKnown` method:

1. Create a `.well-known` directory in your repository root
2. Add `funding.json` to `.well-known/` directory
3. This allows verification via: `https://github.com/Hawksight-AI/semantica/.well-known/funding.json`

### Step 4: Submit to FLOSS/fund Directory

1. Visit: https://dir.floss.fund/submit
2. Enter your `funding.json` manifest URL:
   ```
   https://raw.githubusercontent.com/Hawksight-AI/semantica/main/funding.json
   ```
3. Click "Submit" and "Validate"
4. Your application will be publicly visible on the FLOSS/fund directory

### Step 5: Evaluation Process

- Applications are evaluated quarterly by an internal investment committee
- Factors considered: value, impact, criticality, and innovation
- FLOSS/fund prioritizes existing, widely used, and impactful projects
- If accepted, you'll be contacted to complete paperwork (tax residency documents, etc.)

## Eligibility Criteria

✅ **You Meet These:**
- Open source project (MIT licensed)
- Existing project with development history
- Clear impact and use cases
- No strings attached funding

⚠️ **Considerations:**
- FLOSS/fund prioritizes widely used projects
- They prefer projects with existing user base
- Very new projects with minimal usage may not be considered

## Funding Details

- **Requested Amount**: $50,000 USD per year
- **FLOSS/fund Limits**: 
  - Minimum: $10,000
  - Maximum: $100,000 per year
  - Increments: Multiples of $25,000 after $10,000
- **Your Request**: $50,000 (fits perfectly: $10,000 + $25,000 + $15,000)

## Budget Breakdown (from funding.json)

- **Human Labor**: $25,000 (50%)
  - Lead developer full-time
  - Contributor support and mentorship
  
- **Infrastructure**: $13,000 (26%)
  - Database hosting, cloud CI/CD
  - GPU compute, developer tools
  
- **Community**: $5,000 (10%)
  - Conferences, workshops, engagement
  
- **Research**: $3,000 (6%)
  - Datasets, benchmarks, research
  
- **Operational**: $2,000 (4%)
  - Accounting, compliance, hosting
  
- **Contingency**: $2,000 (4%)
  - Unforeseen needs

## Next Steps

1. **Review funding.json**: Ensure all information is accurate
2. **Commit and Push**: Add funding.json to your repository
3. **Verify URL**: Test that the raw GitHub URL works
4. **Submit**: Go to https://dir.floss.fund/submit and submit your manifest URL
5. **Wait for Evaluation**: Quarterly evaluation cycles
6. **If Accepted**: Complete paperwork and acknowledge funding publicly

## Public Acknowledgment

If funded, FLOSS/fund requests public acknowledgment with a link to their website. Embeddable badges are available.

## Additional Resources

- FLOSS/fund Website: https://floss.fund/
- FLOSS/fund FAQs: https://floss.fund/faq/
- Funding.json Standard: https://fundingjson.org/
- Submission Portal: https://dir.floss.fund/submit

## Notes

- The funding.json file is already created in the repository root
- You may need to adjust the amount or details based on your needs
- Ensure your GitHub repository is public and accessible
- Consider adding the funding.json to `.well-known/` directory for verification

